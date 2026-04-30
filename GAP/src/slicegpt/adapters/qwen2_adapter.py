"""
qwen2_adapter.py（更新版）
兼容 transformers 4.57+ 的新 API：
  - Qwen2DecoderLayer.forward 现在接收 position_embeddings、cache_position
  - Qwen2Attention 返回 (attn_output, attn_weights)，不再返回 present_key_value
  - DynamicCache 在 attention 内部 in-place 更新，模型层只返回 hidden_states tensor
"""

import torch
from torch import FloatTensor, LongTensor, Tensor, matmul
from torch.nn import Linear, Module
from typing import Optional, Tuple
from transformers import PretrainedConfig, PreTrainedTokenizerBase
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Config,
    Qwen2DecoderLayer,
    Qwen2ForCausalLM,
    Qwen2RMSNorm,
)

from slicegpt.model_adapter import LayerAdapter, ModelAdapter


class CompressedQwen2DecoderLayer(Qwen2DecoderLayer):
    """
    带 SliceGPT shortcut_Q 的 Qwen2 解码层（兼容 transformers 4.57+ 新 API）。

    新 API 变化：
      - forward 接收 position_embeddings (cos, sin) 而非 position_ids（RoPE 在模型层统一预算）
      - cache_position 显式传入（避免 DynamicCache.get_seq_length() 触发追踪问题）
      - 返回值从 tuple 改为单个 hidden_states Tensor（模型层直接用返回值迭代）
    """

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[LongTensor] = None,
        past_key_values=None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[LongTensor] = None,
        position_embeddings: Optional[Tuple[Tensor, Tensor]] = None,
        **kwargs,
    ) -> Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        # shortcut_Q：对 residual 做旋转投影（SliceGPT 核心）
        if self.attn_shortcut_Q is not None:
            hidden_states = matmul(residual, self.attn_shortcut_Q) + hidden_states
        else:
            hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        if self.mlp_shortcut_Q is not None:
            hidden_states = matmul(residual, self.mlp_shortcut_Q) + hidden_states
        else:
            hidden_states = residual + hidden_states

        return hidden_states


class Qwen2LayerAdapter(LayerAdapter):
    def __init__(self, layer: Qwen2DecoderLayer) -> None:
        super().__init__()
        self._layer: Qwen2DecoderLayer = layer

    @property
    def layer(self) -> Module:
        return self._layer

    @property
    def hidden_states_args_position(self) -> int:
        return 0

    @property
    def hidden_states_output_position(self) -> int:
        return 0

    def get_first_layernorm(self) -> Module:
        return self.layer.input_layernorm

    def get_second_layernorm(self) -> Module:
        return self.layer.post_attention_layernorm

    def get_attention_inputs(self) -> list[Linear]:
        return [self.layer.self_attn.q_proj, self.layer.self_attn.k_proj, self.layer.self_attn.v_proj]

    def get_attention_output(self) -> Linear:
        return self.layer.self_attn.o_proj

    def get_mlp_inputs(self) -> list[Linear]:
        return [self.layer.mlp.gate_proj, self.layer.mlp.up_proj]

    def get_mlp_output(self) -> Linear:
        return self.layer.mlp.down_proj


class Qwen2ModelAdapter(ModelAdapter):
    def __init__(self, model: Qwen2ForCausalLM) -> None:
        super().__init__()
        self._model: Qwen2ForCausalLM = model

    @property
    def model(self) -> Module:
        return self._model

    @property
    def config(self) -> PretrainedConfig:
        return self._model.config

    @property
    def config_type(self) -> type:
        return Qwen2Config

    @property
    def parallel_blocks(self) -> bool:
        return False

    @property
    def seqlen(self) -> int:
        return self.config.max_position_embeddings

    @property
    def hidden_size(self) -> int:
        return self.config.hidden_size

    @property
    def should_bake_mean_into_linear(self) -> bool:
        return False

    @property
    def original_layer_type(self) -> type:
        return Qwen2DecoderLayer

    @property
    def original_layer_norm_type(self) -> type:
        return Qwen2RMSNorm

    @property
    def layer_adapter_type(self) -> type:
        return Qwen2LayerAdapter

    @property
    def compressed_layer_type(self) -> type:
        return CompressedQwen2DecoderLayer

    @property
    def use_cache(self) -> bool:
        return self.config.use_cache

    @use_cache.setter
    def use_cache(self, value: bool) -> None:
        self.config.use_cache = value

    def compute_output_logits(self, input_ids: Tensor) -> FloatTensor:
        return self.model(input_ids=input_ids).logits

    def convert_layer_to_compressed(self, layer: Module, layer_idx: int | None) -> Module:
        compressed_layer = self.compressed_layer_type(self.config, layer_idx).to(self.config.torch_dtype)
        compressed_layer.load_state_dict(layer.state_dict(), strict=True)
        return compressed_layer

    def get_layers(self) -> list[LayerAdapter]:
        return [self.layer_adapter_type(layer) for layer in self.model.model.layers]

    def get_raw_layer_at(self, index: int) -> Module:
        return self.model.model.layers[index]

    def set_raw_layer_at(self, index: int, new_layer: Module) -> None:
        self.model.model.layers[index] = new_layer

    def get_embeddings(self) -> list[Module]:
        return [self.model.model.embed_tokens]

    def get_pre_head_layernorm(self) -> Module:
        return self.model.model.norm

    def get_lm_head(self) -> Linear:
        return self.model.lm_head

    def post_init(self, tokenizer: PreTrainedTokenizerBase) -> None:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.config.pad_token_id = tokenizer.pad_token_id

    @classmethod
    def _from_pretrained(
        cls,
        model_name: str,
        model_path: str,
        *,
        dtype: torch.dtype = torch.float16,
        local_files_only: bool = False,
        token: str | bool | None = None,
    ) -> "ModelAdapter | None":
        if "qwen" not in model_name.lower():
            return None

        model = Qwen2ForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, token=token, local_files_only=local_files_only
        )
        model.config.torch_dtype = dtype
        return Qwen2ModelAdapter(model)

    @classmethod
    def _from_uninitialized(
        cls,
        model_name: str,
        model_path: str,
        *,
        dtype: torch.dtype = torch.float16,
        local_files_only: bool = False,
        token: str | bool | None = None,
    ) -> "ModelAdapter | None":
        if "qwen" not in model_name.lower():
            return None

        class UninitializedQwen2ForCausalLM(Qwen2ForCausalLM):
            def _init_weights(self, _) -> None:
                pass

        config = Qwen2Config.from_pretrained(
            model_path, torch_dtype=dtype, token=token, local_files_only=local_files_only
        )
        model = UninitializedQwen2ForCausalLM(config)
        model = model.to(dtype=dtype)
        return Qwen2ModelAdapter(model)
