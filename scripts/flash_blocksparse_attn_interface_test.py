import os
import sys
import math
import pickle
from typing import Optional
import pathlib
import torch
import torch.nn as nn
from flash_attn.flash_blocksparse_attn_interface import convert_blockmask
from flash_attn.flash_attn_interface import flash_attn_func
import torch.utils.benchmark as benchmark
import flash_attn_cuda
import torch.nn.functional as F
import traceback


from sparsity_config_square import (
    FixedSparsityConfig,
    DenseSparsityConfig,
    BigBirdSparsityConfig,
    VariableSparsityConfig,
    BSLongformerSparsityConfig,
    LocalSlidingWindowSparsityConfig,
)

from sparsity_config_square import directional_scaling, get_sparsity_config_cls


def collapse_first_n_dims(x, n):
    new_shape = (-1,) + x.shape[n:]
    return x.view(new_shape)


############# Packed Version of the Code ##############

def _flash_blocksparse_attn_forward(
    qkv: torch.Tensor,
    cu_seqlens: torch.IntTensor,
    blockmask: torch.IntTensor,
    dropout_p: float,
    max_s: int,
    softmax_scale: float,
    causal: bool,
    return_softmax: bool,
):
    # the fwd block takes the q, k and v as separate tensors.
    # so we pass them as such
    # # print(qkv.shape)
    q, k, v = qkv[:, 0, :, :], qkv[:, 1, :, :], qkv[:, 2, :, :]
    cu_seqlens_q = cu_seqlens
    cu_seqlens_k = cu_seqlens
    max_seqlen_q, max_seqlen_k = max_s, max_s
    context, softmax_lse, *rest = flash_attn_cuda.fwd_block(
        q,  # (bs * seq_len, num_heads, head_size)
        k,  # (bs * seq_len, num_heads, head_size)
        v,  # (bs * seq_len, num_heads, head_size)
        cu_seqlens_q,  # (bs + 1,)
        cu_seqlens_k,  # (bs + 1,)
        blockmask,  # (seqlen // 256, seqlen // 16), 16 is the row block size, 256 is the column block size
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
        return_softmax,
        None,
    )
    # if context.isnan().any() or softmax_lse.isnan().any():
    #     breakpoint()
    S_dmask = rest[0] if return_softmax else None
    return context, softmax_lse, S_dmask


def _flash_blocksparse_attn_backward(
    dout: torch.Tensor,
    qkv: torch.Tensor,
    out: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    softmax_lse: torch.Tensor,
    cu_seqlens: torch.IntTensor,
    blockmask: torch.IntTensor,
    dropout_p: float,
    max_s: int,
    softmax_scale: float,
    causal: bool
) -> torch.Tensor:
    q, k, v = qkv[:, 0, :, :], qkv[:, 1, :, :], qkv[:, 2, :, :]
    cu_seqlens_q, cu_seqlens_k = cu_seqlens, cu_seqlens
    max_seqlen_q, max_seqlen_k = max_s, max_s
    dout = dout.contiguous()
    print('doing backward on sparse blocks')
    softmax_d = flash_attn_cuda.bwd_block(
        dout,  # (bs * seq_len, num_heads, head_size)
        q,  # (bs * seq_len, num_heads, head_size)
        k,  # (bs * seq_len, num_heads, head_size)
        v,  # (bs * seq_len, num_heads, head_size)
        out,
        softmax_lse,
        dq,
        dk,
        dv,
        cu_seqlens_q,
        cu_seqlens_k,
        blockmask,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
        None
    )
    # if dqkv.isnan().any() or softmax_d.isnan().any():
    #     breakpoint()
    return dq, dk, dv, softmax_d

class FlashBlocksparseAttnFun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, qkv, cu_seqlens, blockmask, dropout_p, max_s, softmax_scale, causal):
        # Save rng_state because the backward pass will regenerate the dropout mask
        rng_state = torch.cuda.get_rng_state() if dropout_p > 0 else None
        if softmax_scale is None:
            softmax_scale = qkv.shape[-1] ** (-0.5)
        context, softmax_lse, S_dmask = _flash_blocksparse_attn_forward(
            qkv, cu_seqlens, blockmask, dropout_p, max_s, softmax_scale, causal=causal,
            return_softmax=False
        )
        ctx.save_for_backward(qkv, context, S_dmask, softmax_lse, cu_seqlens, blockmask, rng_state)
        ctx.dropout_p = dropout_p
        ctx.max_s = max_s
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        return context

    @staticmethod
    def backward(ctx, dout):
        qkv, context, S_dmask, softmax_lse, cu_seqlens, blockmask, rng_state = ctx.saved_tensors
        if rng_state is not None:
            cur_rng_state = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(rng_state)
        # S_dmask is None, temporarily use another tensor just to get it running
        dqkv = torch.empty_like(qkv)
        _flash_blocksparse_attn_backward(
            dout=dout,
            qkv=qkv,
            out=context,
            dq=dqkv[:, 0, :, :],
            dk=dqkv[:, 1, :, :],
            dv=dqkv[:, 2, :, :],
            softmax_lse=softmax_lse,
            cu_seqlens=cu_seqlens,
            blockmask=blockmask,
            dropout_p=ctx.dropout_p,
            max_s=ctx.max_s,
            softmax_scale=ctx.softmax_scale,
            causal=ctx.causal,
        )
        if rng_state is not None:
            torch.cuda.set_rng_state(cur_rng_state)
        return dqkv, None, None, None, None, None, None


# We duplicate code to return both the output and the softmax for testing
# Returning both makes backward a bit slower, so we want to keep using the other version for speed.
class FlashBlocksparseAttnFunWithS(torch.autograd.Function):

    @staticmethod
    def forward(ctx, qkv, cu_seqlens, blockmask, dropout_p, max_s, softmax_scale, causal):
        # Save rng_state because the backward pass is gonna regenerate the dropout mask
        rng_state = torch.cuda.get_rng_state() if dropout_p > 0 else None
        if softmax_scale is None:
            softmax_scale = qkv.shape[-1] ** (-0.5)
        context, softmax_lse, S_dmask = _flash_blocksparse_attn_forward(
            qkv, cu_seqlens, blockmask, dropout_p, max_s, softmax_scale, causal=causal,
            return_softmax=True
        )
        ctx.save_for_backward(qkv, context, S_dmask, softmax_lse, cu_seqlens, blockmask, rng_state)
        ctx.dropout_p = dropout_p
        ctx.max_s = max_s
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        return context, S_dmask, softmax_lse

    @staticmethod
    def backward(ctx, dout, *args):
        qkv, context, S_dmask, softmax_lse, cu_seqlens, blockmask, rng_state = ctx.saved_tensors
        if rng_state is not None:
            cur_rng_state = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(rng_state)
        raise NotImplementedError
        """
        dqkv = _flash_blocksparse_attn_backward(
            dout, qkv, context, S_dmask, softmax_lse, cu_seqlens, blockmask, ctx.dropout_p,
            ctx.max_s, ctx.softmax_scale, ctx.causal
        )
        """
        if rng_state is not None:
            torch.cuda.set_rng_state(cur_rng_state)
        return dqkv, None, None, None, None, None, None


def flash_blocksparse_attn_func(
    qkv: torch.Tensor,
    cu_seqlens: torch.IntTensor,
    blockmask: torch.IntTensor,
    dropout_p: float,
    max_s: int,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    return_attn_probs: bool = False,
    convert_mask: bool = True,
) -> torch.Tensor:
    func = FlashBlocksparseAttnFun if not return_attn_probs else FlashBlocksparseAttnFunWithS
    if convert_mask:
        blockmask = convert_blockmask(blockmask, causal=causal)
    return func.apply(
        qkv,
        cu_seqlens,
        blockmask,
        dropout_p,
        max_s,
        softmax_scale,
        causal,
    )

######### Unit tests start ########


def test_blocksparse_attn_fwd_bwd(inp):
    # This passes okay
    device = torch.device("cuda:1")
    qkv = inp["qkv"].to(device)
    # print(qkv.shape)
    qkv.requires_grad = True
    cu_seqlens = inp["cu_seqlens"].to(device)
    blockmask = inp["blockmask"].to(device)
    dropout_p = inp["dropout_p"]
    max_s = inp["max_s"]
    softmax_scale = inp["softmax_scale"]
    causal = inp["causal"]
    convert_mask = inp.get("convert_mask", False)
    context_layer = flash_blocksparse_attn_func(
        # qkv=collapse_first_n_dims(qkv, 2),
        qkv=qkv,
        cu_seqlens=cu_seqlens,
        blockmask=blockmask,
        dropout_p=dropout_p,
        max_s=max_s,
        softmax_scale=softmax_scale,
        causal=causal,
        return_attn_probs=False,
        convert_mask=convert_mask,
    )
    item = context_layer.sum()
    item.backward()
    # print("Forward and backward executed successfully")


def test_blocksparse_dense_match(sq=None, sparse_config='dense'):
    # # print('Testing blocksparse and dense, on device:', DEVICE)
    # This test case fails right now :/
    bs = 1
    hdim = 128
    sq = sq or 384
    max_sq = sq
    n_heads = 32
    device = torch.device("cuda:1")
    causal = True
    return_attn_probs = False
    def repeat_along_dim(arr, n_times, dim):
        return arr.repeat(*(1 if i != dim else n_times for i in range(arr.ndim)))
    
    Wq = nn.Linear(hdim, hdim, bias=False, device=device, dtype=torch.float16)
    Wk = nn.Linear(hdim, hdim, bias=False, device=device, dtype=torch.float16)
    Wv = nn.Linear(hdim, hdim, bias=False, device=device, dtype=torch.float16)

    def create_query_key_value():
        key = torch.randn(bs, sq, hdim, dtype=torch.float16)
        """
        for ix in range(min(sq, hdim)):
            key[:, ix, ix] = 1
        """
        query = torch.randn(bs, sq, hdim, dtype=torch.float16)
        """
        query[:, 0, 1] = 1.0
        query[:, 1, 0] = 1.0
        for idx in range(2, min(hdim, sq)):
            query[:, idx, idx] = 1.0
        """
        # value = torch.arange(bs * sq * hdim, dtype=torch.float16).reshape((bs, hdim, sq)).transpose(1, 2)
        value = torch.randn((bs, sq, hdim), dtype=torch.float16).reshape((bs, hdim, sq)).transpose(1, 2)
        
        key = Wk(key.to(device=device))
        query = Wq(query.to(device=device))
        value = Wv(value.to(device=device))
        
        query = repeat_along_dim(query.unsqueeze(2), n_heads, 2)
        key = repeat_along_dim(key.unsqueeze(2), n_heads, 2)
        value = repeat_along_dim(value.unsqueeze(2), n_heads, 2)
        qkv = torch.cat([query.unsqueeze(2), key.unsqueeze(2), value.unsqueeze(2)], dim=2)
        # qkv.requires_grad = True
        return qkv.to(device=device, dtype=torch.float16)

    # shape: (bs, sq, 3, nh, hdim)
    qkv = create_query_key_value()
    # shape: (bs, sq, nh, hdim)
    query = qkv[:, :, 0]
    # shape: (bs, sq, nh, hdim)
    key = qkv[:, :, 1]
    # shape: (bs, sq, nh, hdim)
    value = qkv[:, :, 2]
    
    # Using blocksparse interface
    row_block_size = 16
    col_block_size = 128 if hdim > 64 else 256
    
    def create_layout():
        """
            from sparsity_config_square import (
                FixedSparsityConfig,
                DenseSparsityConfig,
                BigBirdSparsityConfig,
                VariableSparsityConfig,
                BSLongformerSparsityConfig,
                LocalSlidingWindowSparsityConfig,
            )
        """
        if sparse_config in ['localslide', 'bigbird', 'longformer']:
            config = get_sparsity_config_cls(sparse_config)(1, block=col_block_size, attention='unidirectional' if causal else 'bidirectional', num_sliding_window_blocks=min(sq//col_block_size, 7))
        else:
            config = get_sparsity_config_cls(sparse_config)(1, block=col_block_size, attention='unidirectional' if causal else 'bidirectional')
        layout = directional_scaling(config.make_layout(max_sq), col_block_size//row_block_size)
        layout = layout[:(sq + row_block_size - 1) // row_block_size, :(sq + col_block_size - 1) // col_block_size].to(dtype=torch.int32, device=device)
        return layout
    
    # shape: (sq // row_block_size, sq // col_block_size)
    layout = create_layout()
    sparse_blockmask = convert_blockmask(layout, causal, row_block_size, col_block_size)

    assert query.shape[2] == key.shape[2] == value.shape[2] == n_heads
    # shape: (bs, nh, sq, sq)
    dense_logits = torch.einsum("bsnh,bqnh -> bnsq", query, key) * 1.0 / math.sqrt(hdim)
    if causal:
        causal_masking = torch.full_like(dense_logits, -10000).to(dtype=torch.float16).triu(1)
        dense_logits = dense_logits + causal_masking
    
    # scale blockmask to trivial implement
    dense_blockmask = -10000*(1-layout.repeat_interleave(row_block_size, -2).repeat_interleave(col_block_size, -1)[:sq, :sq])
    # print(dense_blockmask.shape)
    dense_logits = dense_logits + dense_blockmask

    # shape: (bs, nh, sq, sq)
    dense_scores = torch.softmax(dense_logits, dim=-1)
    
    # shape: (bs, nh, sq, hdim)
    dense_values = torch.einsum("bnst,btnd -> bsnd", dense_scores, value)  # XXX

    # Using FlashAttention Interface
    bs, sq = qkv.size(0), qkv.size(1)
    cu_seqlens = torch.arange(0, (bs + 1)*sq, sq, device=device, dtype=torch.int32)
    if return_attn_probs:
        flash_attn_dense_values, softmax_lse, S_dmask = flash_attn_func(
            qkv=collapse_first_n_dims(qkv.clone(), 2),
            cu_seqlens=cu_seqlens,
            dropout_p=0.0,
            max_s=sq,
            softmax_scale=None,
            causal=causal,
            return_attn_probs=True,
        )
        # print(softmax_lse)
        # print(softmax_lse.shape)
        # print(S_dmask)
        # print(S_dmask.shape)
    else:
        flash_attn_dense_values = flash_attn_func(
            qkv=collapse_first_n_dims(qkv.clone(), 2),
            cu_seqlens=cu_seqlens,
            dropout_p=0.0,
            max_s=sq,
            softmax_scale=None,
            causal=causal,
            return_attn_probs=False,
        )
    flash_attn_dense_values = flash_attn_dense_values.reshape((bs, sq, n_heads, hdim))
    
    # print(blockmask)
    # exit(0)
    
    # if hdim > 64 or sq > 128:
    if return_attn_probs:
        flash_attn_sparse_values, S_dmask, softmax_lse = flash_blocksparse_attn_func(
            qkv=collapse_first_n_dims(qkv.clone(), 2),
            cu_seqlens=cu_seqlens,
            blockmask=sparse_blockmask,
            dropout_p=0.0,
            max_s=sq,
            softmax_scale=None,
            causal=causal,
            return_attn_probs=True,
            convert_mask=False,
        )
    else:
        flash_attn_sparse_values = flash_blocksparse_attn_func(
            qkv=collapse_first_n_dims(qkv.clone(), 2),
            cu_seqlens=cu_seqlens,
            blockmask=sparse_blockmask,
            dropout_p=0.0,
            max_s=sq,
            softmax_scale=None,
            causal=causal,
            return_attn_probs=False,
            convert_mask=False,
        )
    flash_attn_sparse_values = flash_attn_sparse_values.reshape((bs, sq, n_heads, hdim))
    
    def get_statistics(x, x_name):
        x_shape = x.shape
        x_numel = x.numel()
        x_max = x.max()
        x_min = x.min()
        x_mean = x.mean()
        x_std = x.std()
        x_abs_max = x.abs().max()
        x_norm = x.to(dtype=torch.float64).norm()
        print(f'Statistics of {x_name}:\nShape: \t{x_shape}\nNumel: \t{x_numel}\nMax: \t{x_max}\nAMax: \t{x_abs_max}\nMin: \t{x_min}\nMean: \t{x_mean}\nStd: \t{x_std}\nNorm: \t{x_norm}\n')
        
    def get_error_statistics(x, y, x_name, y_name, verbose=False):
        if verbose:
            get_statistics(x, x_name)
            get_statistics(y, y_name)
        get_statistics(x - y, f'error between {x_name} and {y_name}')
        print('===================================================================')
        print()
        
    def check_flashattnsparse():
        for i in range(flash_attn_sparse_values.shape[1]):
            if flash_attn_sparse_values[0, i, 0, :].to(dtype=torch.float64).norm() != 0:
                print(f'The first row of flash_attn_sparse_values which contains nonzero is {i}')
                break
        
    def test_flashattndense_flashattnsparse():
        get_error_statistics(flash_attn_dense_values, flash_attn_sparse_values, 'flash_attn_dense', 'flash_attn_sparse', verbose=True)
        print(flash_attn_dense_values)
        print(flash_attn_dense_values.shape)
        print(flash_attn_sparse_values)
        print(flash_attn_sparse_values.shape)
        print(flash_attn_dense_values - flash_attn_sparse_values)
        print()
        assert torch.allclose(flash_attn_sparse_values, flash_attn_dense_values)
        if torch.allclose(flash_attn_sparse_values, flash_attn_dense_values):
            print('[OK] Flash Attention gives the same result between sparse and dense!')
            ...
        else:
            # not expected to happen
            print('[FAIL] Flash Attention gives different results between sparse and dense!')
            ...
            
    def test_dense_flashattndense():
        get_error_statistics(dense_values, flash_attn_dense_values, 'dense', 'flash_attn_dense', verbose=True)
        # print(dense_values - flash_attn_dense_values)
        # assert torch.allclose(dense_values, flash_attn_dense_values)
            
    def test_dense_flashattnsparse():
        get_error_statistics(dense_values, flash_attn_sparse_values, 'dense', 'flash_attn_sparse', verbose=True)
        print(dense_values - flash_attn_sparse_values)
        assert torch.allclose(dense_values, flash_attn_sparse_values)
        
    
    dense_values.sum().backward(retain_graph=True)
    dk = Wk.weight.grad.clone().unsqueeze(0)
    dq = Wq.weight.grad.clone().unsqueeze(0)
    dv = Wv.weight.grad.clone().unsqueeze(0)
    dqkv = torch.cat([dq, dk, dv], dim=0)
    # print(dqkv)
    # get_statistics(dqkv, 'dqkv')
    # exit(0)
    
    # print(dqkv.shape)
    # dqkv = torch.cat([dq.unsqueeze(1), dk.unsqueeze(1), dv.unsqueeze(1)], dim=1)
    # print(Wk.weight.grad, Wq.weight.grad, Wv.weight.grad)
    Wk.zero_grad()
    Wq.zero_grad()
    Wv.zero_grad()
    flash_attn_sparse_values.sum().backward()
    sparse_dk = Wk.weight.grad.clone().unsqueeze(0)
    sparse_dq = Wq.weight.grad.clone().unsqueeze(0)
    sparse_dv = Wv.weight.grad.clone().unsqueeze(0)
    sparse_dqkv = torch.cat([sparse_dq, sparse_dk, sparse_dv], dim=0)
    # print(sparse_dqkv.shape)
    # print(dqkv - sparse_dqkv)
    get_error_statistics(dqkv, sparse_dqkv, 'pytorch_dqkv', 'flashattnsparse_dqkv', verbose=True)
    # print(dqkv)
    # print(sparse_dqkv)
    # get_statistics(sparse_dqkv, 'sparse_dqkv')
    get_statistics((dqkv - sparse_dqkv)[0], 'error_dq')
    get_statistics((dqkv - sparse_dqkv)[1], 'error_dk')
    get_statistics((dqkv - sparse_dqkv)[2], 'error_dv')
    # print(sparse_dqkv[0] - sparse_dqkv[1])
    # print((dqkv - sparse_dqkv)[1])

    # print(Wk.weight.grad, Wq.weight.grad, Wv.weight.grad)
        
    # print('???')
    # test_dense_flashattnsparse()
    # test_flashattndense_flashattnsparse()
    # check_flashattnsparse()
    
    # assert torch.allclose(flash_attn_sparse_values, flash_attn_dense_values)

    # test_dense_flashattndense()
    # # print(flash_attn_dense_values)
    # test_dense_flashattnsparse()
    # import pdb; pdb.set_trace()


def measure_func_time(fn, *inputs, repeats = 10, desc='', verbose=True, **kwinputs):
    if verbose:
        # print(desc, '- Forward pass')
        ...
    t = benchmark.Timer(
            stmt='fn(*inputs, **kwinputs)',
            globals={'fn': fn, 'inputs': inputs, 'kwinputs': kwinputs},
            num_threads=torch.get_num_threads(),
            )
    m = t.timeit(repeats)
    if verbose:
        # print(m)
        ...
    return t, m


def load_inp(seqlen):
    fixture_path = f'/remote-home/qhgao/Workspace/Projects/MainUniverse/artworks/flash-attention-modify/input_{seqlen}_variable.pkl'
    assert os.path.exists(fixture_path)
    inp = pickle.load(open(fixture_path, 'rb'))
    return inp


def test_test_blocksparse_attn_fwd_bwd(seqlen=2048):
    # print(f'Sequence Length: {seqlen}')
    inp = load_inp(seqlen=seqlen)
    timer, time = measure_func_time(test_blocksparse_attn_fwd_bwd, inp=inp, repeats=10)
    # print(time)
    # print()


if __name__ == "__main__":
    """
    for seqlen in [2048, 4096, 8192, 16384, 32768, 65536, 131072]:
        test_test_blocksparse_attn_fwd_bwd(seqlen)
    """
    
    # test_test_blocksparse_attn_fwd_bwd(131072)
    # test_blocksparse_dense_match()
    # exit(0)
    for sparsity_config in ['dense', 'fixed', 'variable', 'bigbird', 'longformer', 'localslide']:
        print(f'testing sparsity config {sparsity_config}\n')
        for seqlen in [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072][::1]:
            exp_raised = False
            try:
                print(f'testing sequence length {seqlen}')
                f = open(f'pytorch_flashattnsparse_bwd_numericaltest_{sparsity_config}_{seqlen}_headnum32.log', 'w')
                sys.stdout = f
                sys.stderr = f
                test_blocksparse_dense_match(seqlen, sparsity_config)
            except Exception as e:
                print(traceback.format_exc())
                print(f'[{type(e).__name__}] {e}')
                exp_raised = True
            f.close()
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            if exp_raised:
                print(f'failed at sequence length {seqlen}')
                break
            else:
                print('finished test')
        print()
    exit(0)
    for sq in range(1, 2049):
        try:
            test_blocksparse_dense_match(sq)
        except AssertionError:
            print(f'Failed at sequence length {sq}')
        else:
            print(f'Passed at sequence length {sq}')