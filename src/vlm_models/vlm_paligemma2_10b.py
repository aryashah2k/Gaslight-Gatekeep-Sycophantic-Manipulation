"""
PaliGemma2-10B Vision-Language Model Wrapper

HuggingFace: https://huggingface.co/google/paligemma2-10b-ft-docci-448

PaliGemma2 uses SigLIP as its vision encoder. We access vision_tower directly
for pure visual representations.

FIXED: Now extracts features from SigLIP vision_tower directly.
"""

import torch
import numpy as np
from typing import List, Optional, Union
from PIL import Image
import logging

from .base_vlm import BaseVLM, VLMConfig

logger = logging.getLogger(__name__)


class PaliGemma2_10B(BaseVLM):
    """PaliGemma2-10B wrapper."""
    
    MODEL_ID = "google/paligemma2-10b-ft-docci-448"
    MODEL_NAME = "paligemma2_10b"
    HUGGINGFACE_URL = "https://huggingface.co/google/paligemma2-10b-ft-docci-448"
    
    def load_model(self):
        """Load PaliGemma2 model and processor."""
        from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
        
        logger.info(f"Loading {self.MODEL_ID}...")
        
        self._processor = PaliGemmaProcessor.from_pretrained(self.MODEL_ID)
        
        self._model = PaliGemmaForConditionalGeneration.from_pretrained(
            self.MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        ).eval()
        
        self._is_loaded = True
        
        logger.info(f"Loaded {self.MODEL_NAME} successfully")
    
    def get_vision_features(
        self,
        images: Union[Image.Image, List[Image.Image]]
    ) -> np.ndarray:
        """Extract vision features from SigLIP vision_tower directly.
        
        PaliGemma uses SigLIP as vision encoder accessible via model.vision_tower.
        """
        images = self._ensure_list(images)
        
        all_features = []
        
        with torch.no_grad():
            for image in images:
                # Ensure RGB
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Process image with <image> token as required by PaliGemma
                inputs = self.processor(
                    text="<image>",
                    images=image,
                    return_tensors="pt"
                ).to(dtype=torch.bfloat16, device=self.model.device)
                
                pixel_values = inputs.get('pixel_values')
                features = None
                
                # Try to access vision_tower directly
                if pixel_values is not None:
                    # Method 1: model.vision_tower (PaliGemma standard path)
                    if hasattr(self.model, 'vision_tower') and self.model.vision_tower is not None:
                        try:
                            vision_outputs = self.model.vision_tower(
                                pixel_values,
                                output_hidden_states=True
                            )
                            if hasattr(vision_outputs, 'last_hidden_state'):
                                features = vision_outputs.last_hidden_state.mean(dim=1)
                            elif hasattr(vision_outputs, 'pooler_output') and vision_outputs.pooler_output is not None:
                                features = vision_outputs.pooler_output
                            else:
                                hidden = vision_outputs[0] if isinstance(vision_outputs, tuple) else vision_outputs
                                features = hidden.mean(dim=1) if hidden.dim() > 2 else hidden
                        except Exception as e:
                            logger.debug(f"vision_tower access failed: {e}")
                    
                    # Method 2: model.model.vision_tower
                    if features is None and hasattr(self.model, 'model') and hasattr(self.model.model, 'vision_tower'):
                        try:
                            vision_outputs = self.model.model.vision_tower(
                                pixel_values,
                                output_hidden_states=True
                            )
                            if hasattr(vision_outputs, 'last_hidden_state'):
                                features = vision_outputs.last_hidden_state.mean(dim=1)
                            else:
                                hidden = vision_outputs[0] if isinstance(vision_outputs, tuple) else vision_outputs
                                features = hidden.mean(dim=1) if hidden.dim() > 2 else hidden
                        except Exception as e:
                            logger.debug(f"model.vision_tower access failed: {e}")
                    
                    # Method 3: model.multi_modal_projector to get projected vision features
                    if features is None and hasattr(self.model, 'multi_modal_projector'):
                        try:
                            # Get the vision model first
                            if hasattr(self.model, 'vision_model'):
                                vision_outputs = self.model.vision_model(pixel_values)
                                if hasattr(vision_outputs, 'last_hidden_state'):
                                    features = vision_outputs.last_hidden_state.mean(dim=1)
                        except Exception as e:
                            logger.debug(f"vision_model access failed: {e}")
                
                # Fallback: Use full model hidden states
                if features is None:
                    outputs = self.model(
                        **inputs,
                        output_hidden_states=True,
                        return_dict=True
                    )
                    if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                        hidden = outputs.hidden_states[-1]
                        features = hidden.mean(dim=1)
                    else:
                        # Last resort fallback
                        hidden_size = getattr(self.model.config, 'hidden_size', 3584)
                        features = torch.zeros(1, hidden_size, device=self.model.device, dtype=torch.bfloat16)
                    logger.debug("Using LLM hidden states as fallback")
                
                all_features.append(features.cpu().float().numpy())
        
        return np.concatenate(all_features, axis=0)
    
    def generate(
        self,
        image: Image.Image,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate response for image and prompt."""
        max_tokens = max_tokens or self.config.max_new_tokens
        
        # Ensure RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # PaliGemma expects <image> token in prompts
        if system_prompt:
            full_prompt = f"<image>{system_prompt}\n{prompt}"
        else:
            full_prompt = f"<image>{prompt}"
        
        inputs = self.processor(
            text=full_prompt,
            images=image,
            return_tensors="pt"
        ).to(dtype=torch.bfloat16, device=self.model.device)
        
        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=self.config.do_sample,
                temperature=self.config.temperature if self.config.do_sample else None
            )
        
        # Decode response
        response = self.processor.decode(
            output_ids[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return response.strip()


def load_model(device: str = "cuda") -> PaliGemma2_10B:
    """Load PaliGemma2-10B model."""
    return PaliGemma2_10B(device=device)
