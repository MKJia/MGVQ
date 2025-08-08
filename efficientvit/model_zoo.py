from typing import Callable, Optional

from huggingface_hub import PyTorchModelHubMixin

from efficientvit.models.tokenizer.mgvq import MGVQ, MGVQConfig, mgvq_f8c32, mgvq_f16c32, mgvq_f32c32

__all__ = ["create_mgvq_model_cfg", "MGVQ_HF"]


REGISTERED_DCAE_MODEL: dict[str, tuple[Callable, Optional[str]]] = {
    "mgvq-f8c32": (mgvq_f8c32, None),
    "mgvq-f16c32": (mgvq_f16c32, None),
    "mgvq-f32c32": (mgvq_f32c32, None),
}


def create_mgvq_model_cfg(name: str, codebook_size: int, codebook_groups: int, pretrained_path: Optional[str] = None) -> MGVQConfig:
    assert name in REGISTERED_DCAE_MODEL, f"{name} is not supported"
    mgvq_cls, default_pt_path = REGISTERED_DCAE_MODEL[name]
    pretrained_path = default_pt_path if pretrained_path is None else pretrained_path
    model_cfg = mgvq_cls(name, pretrained_path, codebook_size, codebook_groups)
    return model_cfg


class MGVQ_HF(MGVQ, PyTorchModelHubMixin):
    def __init__(self, args):
        model_name = args.vq_model
        codebook_size = args.codebook_size
        codebook_groups = args.codebook_groups
        cfg = create_mgvq_model_cfg(model_name, codebook_size, codebook_groups)
        MGVQ.__init__(self, cfg)

