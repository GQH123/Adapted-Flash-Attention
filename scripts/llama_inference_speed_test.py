# Copyright (c) 2023, Tri Dao.

# To run the huggingface implementation, we first need to convert the weights:
# https://github.com/huggingface/transformers/pull/21955
# python -m transformers.models.llama.convert_llama_weights_to_hf --input_dir $CHECKPOINT_DIR/llama --model_size 7B --output_dir $CHECKPOINT_DIR/llama/7B-hf
# and repeat for 13B, 30B, 65B

print('loaded')

import os
current_device = '2'
print('Setting CUDA_VISIBLE_DEVICES to:', current_device)
os.environ['CUDA_VISIBLE_DEVICES'] = current_device


import time
from pathlib import Path

current_dir = Path(__file__).parent.absolute()
print(current_dir)

import torch
import pytest

from einops import rearrange

import torch.utils.benchmark as benchmark

from transformers import LlamaTokenizer, LlamaConfig, LlamaTokenizerFast
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from flash_attn.models.gpt import GPTLMHeadModel
from flash_attn.models.llama import remap_state_dict_meta_llama, llama_config_to_gpt2_config, remap_state_dict_hf_llama
from flash_attn.utils.distributed import all_gather_raw
from flash_attn.utils.pretrained import state_dict_from_pretrained
from flash_attn.utils.generation import update_graph_cache

from sparsity_config_square import (
    FixedSparsityConfig,
    DenseSparsityConfig,
    BigBirdSparsityConfig,
    VariableSparsityConfig,
    BSLongformerSparsityConfig,
    LocalSlidingWindowSparsityConfig,
)

sparse_config_cls_list = [
    FixedSparsityConfig,
    DenseSparsityConfig,
    BigBirdSparsityConfig,
    VariableSparsityConfig,
    BSLongformerSparsityConfig,
    LocalSlidingWindowSparsityConfig
]

def benchmark_forward(fn, *inputs, repeats = 10, desc='', verbose=True, **kwinputs):
    if verbose:
        print(desc, '- Forward pass')
    t = benchmark.Timer(
            stmt='fn(*inputs, **kwinputs)',
            globals={'fn': fn, 'inputs': inputs, 'kwinputs': kwinputs},
            num_threads=torch.get_num_threads(),
            )
    m = t.timeit(repeats)
    if verbose:
        print(m)
    return t, m

def generate(model, inputs, **generate_kwargs):
  return model.generate(inputs, **generate_kwargs)  # （1, 2048）


dtype = torch.float16
device = 'cuda:0'
torch.manual_seed(728)
batch_size = 10

for seqlen in [2048, 4096, 8192, 16384, 32768, 65536, 131072]:  # full attention may only applied to 16K and below
    print(seqlen)
    config = LlamaConfig(hidden_size=768, intermediate_size=None,
                         num_attention_heads=12,
                         num_hidden_layers=12,
                         rms_norm_eps=1e-05,
                         use_cache=False,
                         max_position_embeddings=seqlen) # TODO
    config = llama_config_to_gpt2_config(config)

    config.use_flash_attn = True
    config.fused_bias_fc = True
    config.fused_mlp = False  # We don't have fused GatedMLP yet
    config.fused_dropout_add_ln = True
    config.residual_in_fp32 = True
    # block sparse
    config.max_seq_length = seqlen
    # config.sparsity_config = sparse_config_cls_list[int(current_device)-2](1, 64)
    config.sparsity_config = VariableSparsityConfig(1, 64)
    # config.sparsity_config = LocalSlidingWindowSparsityConfig(1, 64, attention='unidirectional', num_sliding_window_blocks=3)

    print("config:\n", config.sparsity_config.__class__.__name__)

    tokenizer = LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")
    tokenizer.pad_token = tokenizer.eos_token
    eos_token_id = tokenizer.eos_token_id

    model = GPTLMHeadModel(config, device=device, dtype=dtype)
    print(model)
    print(f'Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    model.eval()

    input_ids = torch.randint(0, config.vocab_size, (batch_size, seqlen), dtype=torch.long,
                              device=device)

    timer, time = benchmark_forward(generate, inputs=input_ids, model=model,
                                 repeats=10, desc='Llama Generate',
                                 max_length=seqlen, fused_ft_kernel=True,
                                 return_dict_in_generate=True, output_scores=True, timing=True)
    print(time)

    # print('Without CUDA graph')
    # torch.cuda.synchronize()
    # start = time.time()
    # out = model.generate(input_ids=input_ids, max_length=seqlen,
    #                      eos_token_id=eos_token_id, fused_ft_kernel=True,
    #                      return_dict_in_generate=True, output_scores=True, timing=True)
    # torch.cuda.synchronize()
    # print(f'Prompt processing + decoding time: {(time.time() - start) * 1000:.0f}ms')

    del model

