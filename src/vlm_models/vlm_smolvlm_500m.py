"""
SmolVLM-500M Vision-Language Model Wrapper

HuggingFace: https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct

SmolVLM-500M is a larger variant of SmolVLM using SigLIP for vision encoding.
Same architecture as 256M but with more parameters.
"""

from .vlm_smolvlm_256m import SmolVLM500M

# Re-export for consistency with other modules
def load_model(device: str = "cuda") -> SmolVLM500M:
    """Load SmolVLM-500M model."""
    return SmolVLM500M(device=device)
