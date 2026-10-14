from dataclasses import dataclass, fields, is_dataclass
from typing import Tuple
import yaml


@dataclass(frozen=True)
class MaskingConfig:
    context_scale: Tuple[float, float] = (0.85, 1.0)
    target_scale: Tuple[float, float] = (0.15, 0.2)
    target_aspect_ratio: Tuple[float, float] = (0.5, 1.5)
    num_target_blocks: int = 4


@dataclass(frozen=True)
class EncoderConfig:
    embedding_size: int = 256
    num_heads: int = 4
    num_layers: int = 6
    hidden_size: int = 512

    def __post_init__(self):
        if self.embedding_size % self.num_heads:
            raise ValueError(f"embedding_size ({self.embedding_size}) must be divisible by num_heads ({self.num_heads})")


@dataclass(frozen=True)
class PredictorConfig:
    embedding_size: int = 128
    num_heads: int = 4
    num_layers: int = 3
    hidden_size: int = 256

    def __post_init__(self):
        if self.embedding_size % self.num_heads:
            raise ValueError(f"embedding_size ({self.embedding_size}) must be divisible by num_heads ({self.num_heads})")


@dataclass(frozen=True)
class ModelConfig:
    image_shape: Tuple[int, int, int] = (3, 32, 32)
    patch_shape: Tuple[int, int]      = (4, 4)
    context_learning_rate: float      = 0.001
    predictor_learning_rate: float    = 0.001
    model_learning_rate: float        = 0.001
    momentum: float                   = 0.996
    masking: MaskingConfig     = MaskingConfig()
    encoder: EncoderConfig     = EncoderConfig()
    predictor: PredictorConfig = PredictorConfig()

    @property
    def num_patches(self) -> int:
        return (self.image_shape[1] // self.patch_shape[0]) * (self.image_shape[2] // self.patch_shape[1])


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int         = 128
    learning_rate: int      = 1e-3
    num_epochs: int         = 50
    save_metrics_every: int = 20


@dataclass(frozen=True)
class GlobalConfig:
    seed: int = 13
    device: str = "mps"
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()

    @classmethod
    def from_yaml(cls, path):
        with open(path, encoding="utf-8") as f:
            return _build(cls, yaml.safe_load(f) or {})


def _build(cls, data: dict):
    known   = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(
            f"{cls.__name__} : unknown keys : {sorted(unknown)}"
            f"(expexted keys : {sorted(known)})"
        )
    kwargs = {}
    for name, value in data.items():
        field_type = known[name].type
        kwargs[name] = _build(field_type, value) if is_dataclass(field_type) else value
    return cls(**kwargs)
