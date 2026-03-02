"""
System and model configurations
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class ModelConfig:
    """LLM model configuration"""
    name: str
    num_layers: int
    num_heads: int
    head_dim: int
    dtype_bytes: int = 2  # fp16
    
    @property
    def kv_bytes_per_token(self) -> float:
        """KV cache size per token in bytes"""
        # KV cache = 2 (K and V) * layers * heads * head_dim * dtype
        return 2 * self.num_layers * self.num_heads * self.head_dim * self.dtype_bytes
    
    @property
    def kv_mb_per_token(self) -> float:
        return self.kv_bytes_per_token / 1024 / 1024


# Predefined model configs
MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "gpt2": ModelConfig("gpt2", 12, 12, 64),
    "gpt2-xl": ModelConfig("gpt2-xl", 48, 25, 64),
    "llama-7b": ModelConfig("llama-7b", 32, 32, 128),
    "llama-13b": ModelConfig("llama-13b", 40, 40, 128),
    "llama-70b": ModelConfig("llama-70b", 80, 64, 128),
}


@dataclass 
class SystemConfig:
    """System configuration"""
    gpu_memory_mb: float = 24 * 1024  # 24GB
    model_memory_mb: float = 14 * 1024  # Model weights
    
    prefill_time_per_token_ms: float = 0.5
    decode_time_per_token_ms: float = 10.0
    
    max_batch_size: int = 32
    
    @property
    def available_kv_memory_mb(self) -> float:
        return self.gpu_memory_mb - self.model_memory_mb


def get_model_config(name: str) -> ModelConfig:
    if name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {name}")
    return MODEL_CONFIGS[name]
