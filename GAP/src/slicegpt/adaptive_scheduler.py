"""
自适应非均匀切片调度器

为每层提供不同的保留维度数，替代 ConstSlicingScheduler 的均匀分配。
"""

from .slicing_scheduler import SlicingScheduler


class AdaptiveSlicingScheduler(SlicingScheduler):

    def __init__(self, layer_dimensions: list[int]):
        super().__init__(do_slice_head=True)
        self.layer_dimensions = layer_dimensions

    def setup(self, *, hidden_size: int, layers_num: int, parallel_blocks: bool) -> None:
        if len(self.layer_dimensions) != layers_num:
            raise ValueError(f"维度列表长度 ({len(self.layer_dimensions)}) != 模型层数 ({layers_num})")
        super().setup(hidden_size=hidden_size, layers_num=layers_num, parallel_blocks=parallel_blocks)

    def _get_input_embedding_dimensions(self) -> dict[int, int]:
        dim = self.layer_dimensions[0]
        return {0: dim, 1: dim}

    def _get_attention_input_dimension(self, idx: int) -> int:
        return self.layer_dimensions[idx - 1] if idx > 0 else self.layer_dimensions[0]

    def _get_attention_output_dimension(self, idx: int) -> int:
        return self.layer_dimensions[idx]

    def _get_mlp_input_dimension(self, idx: int) -> int:
        return self.layer_dimensions[idx]

    def _get_mlp_output_dimension(self, idx: int) -> int:
        return self.layer_dimensions[idx]

    def _get_head_dimension(self) -> int:
        return self.layer_dimensions[-1]
