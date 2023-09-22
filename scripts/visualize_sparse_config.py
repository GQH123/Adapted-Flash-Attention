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

from sparsity_config_square import directional_scaling
from visualize_matrix import visualize_matrix


def visualize_sparse_config(head_dim=128, seqlen=1024):
    for config_cls in sparse_config_cls_list:
        cls_name = config_cls.__name__
        print(f"Visualizing {cls_name}...")
        b_r = 16
        b_c = 128 if head_dim > 64 else 256
        config = config_cls(1, block=b_c)
        blockmask = config.make_layout(seqlen)
        blockmask = directional_scaling(blockmask, b_c//b_r)
        visualize_matrix(blockmask, path=f'{cls_name}_seq{seqlen}_hdim{head_dim}.png', xlabels=list(range(seqlen//b_c)), ylabels=list(range(seqlen//b_r)), dpi=1000, title=f'{cls_name} Sparsity Pattern', figsize=(16, 16))


def visualize_sparse_config_LocalSlidingWindowSparsityConfig(head_dim=128, seqlen=2048, num_sliding_window_blocks=3):
    config_cls = LocalSlidingWindowSparsityConfig
    cls_name = config_cls.__name__
    print(f"Visualizing {cls_name} with sliding window size {num_sliding_window_blocks}...")
    b_r = 16
    b_c = 128 if head_dim > 64 else 256
    config = config_cls(1, block=b_c, num_sliding_window_blocks=num_sliding_window_blocks)
    blockmask = config.make_layout(seqlen)
    blockmask = directional_scaling(blockmask, b_c//b_r)
    visualize_matrix(blockmask, path=f'{cls_name}_seq{seqlen}_hdim{head_dim}_slide{num_sliding_window_blocks}.png', xlabels=list(range(seqlen//b_c)), ylabels=list(range(seqlen//b_r)), dpi=1000, title=f'{cls_name} Sparsity Pattern', figsize=(16, 16), fontsize='xx-small')


if __name__ == '__main__':
    # visualize_sparse_config()
    for i in range(1, 17, 2):
        visualize_sparse_config_LocalSlidingWindowSparsityConfig(num_sliding_window_blocks=i)