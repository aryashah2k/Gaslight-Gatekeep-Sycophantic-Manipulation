"""
LFM2-VL-450M Vision-Language Model Wrapper

HuggingFace: https://huggingface.co/LiquidAI/LFM2-VL-450M

LFM2-VL uses SigLIP2 NaFlex as vision encoder (86M params, 768 hidden size).
The vision encoder requires pixel_attention_mask and spatial_shapes from the processor.

CORRECT IMPLEMENTATION: Passes all required arguments to vision_tower.
"""

import torch
import numpy as np
from typing import List, Optional, Union
from PIL import Image
import logging

from .base_vlm import BaseVLM, VLMConfig

logger = logging.getLogger(__name__)


class LFM2VL8B(BaseVLM):
    """LFM2-VL-450M wrapper."""
    
    MODEL_ID = "LiquidAI/LFM2-VL-450M"
    MODEL_NAME = "lfm2vl_8b"
    HUGGINGFACE_URL = "https://huggingface.co/LiquidAI/LFM2-VL-450M"
    VISION_HIDDEN_SIZE = 768  # SigLIP2 NaFlex 86M
    
    def load_model(self):
        """Load LFM2-VL model and processor."""
        from transformers import AutoProcessor, AutoModelForImageTextToText
        
        logger.info(f"Loading {self.MODEL_ID}...")
        
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        self._processor = AutoProcessor.from_pretrained(self.MODEL_ID)
        
        self._model.eval()
        self._is_loaded = True
        
        # Validate vision encoder access
        if not hasattr(self.model, 'vision_tower'):
            raise RuntimeError(f"{self.MODEL_NAME}: model.vision_tower not found")
        
        logger.info(f"Loaded {self.MODEL_NAME} successfully (vision_tower: {type(self.model.vision_tower).__name__})")
    
    def get_vision_features(
        self,
        images: Union[Image.Image, List[Image.Image]]
    ) -> np.ndarray:
        """Extract vision features from SigLIP2 vision encoder.
        
        SigLIP2 NaFlex requires pixel_values, pixel_attention_mask, and spatial_shapes.
        We get these from the processor and pass them to vision_tower.
        """
        images = self._ensure_list(images)
        all_features = []
        
        with torch.no_grad():
            for i, image in enumerate(images):
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Prepare inputs - processor returns all required tensors
                conversation = [{
                    "role": "user",
                    "content": [{"type": "image", "image": image}]
                }]
                
                inputs = self.processor.apply_chat_template(
                    conversation,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                    tokenize=True
                ).to(self.model.device)
                
                pixel_values = inputs.get('pixel_values')
                pixel_attention_mask = inputs.get('pixel_attention_mask')
                spatial_shapes = inputs.get('spatial_shapes')
                
                if pixel_values is None:
                    raise RuntimeError(f"{self.MODEL_NAME}: processor did not return pixel_values")
                
                # Call vision_tower with all required arguments
                vision_tower = self.model.vision_tower
                
                try:
                    # SigLIP2VisionModel requires: pixel_values, pixel_attention_mask, spatial_shapes
                    vision_outputs = vision_tower(
                        pixel_values=pixel_values,
                        pixel_attention_mask=pixel_attention_mask,
                        spatial_shapes=spatial_shapes
                    )
                    
                    if hasattr(vision_outputs, 'last_hidden_state'):
                        # Mean pool over spatial tokens
                        hidden_state = vision_outputs.last_hidden_state
                        if pixel_attention_mask is not None:
                            # Use attention mask for proper pooling
                            mask = pixel_attention_mask.unsqueeze(-1).float()
                            features = (hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                        else:
                            features = hidden_state.mean(dim=1)
                    elif hasattr(vision_outputs, 'pooler_output') and vision_outputs.pooler_output is not None:
                        features = vision_outputs.pooler_output
                    else:
                        # Fallback: treat output as tensor
                        hidden = vision_outputs[0] if isinstance(vision_outputs, tuple) else vision_outputs
                        features = hidden.mean(dim=1) if hidden.dim() > 2 else hidden
                    
                    logger.debug(f"Image {i}: Extracted via vision_tower, shape {features.shape}")
                    
                except Exception as e:
                    raise RuntimeError(f"{self.MODEL_NAME}: vision_tower extraction failed: {e}")
                
                # Validate features
                if torch.isnan(features).any():
                    raise RuntimeError(f"{self.MODEL_NAME}: NaN values in extracted features for image {i}")
                if torch.isinf(features).any():
                    raise RuntimeError(f"{self.MODEL_NAME}: Inf values in extracted features for image {i}")
                if features.abs().sum() == 0:
                    raise RuntimeError(f"{self.MODEL_NAME}: All-zero features for image {i}")
                
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
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        conversation = []
        
        if system_prompt:
            conversation.append({
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            })
        
        conversation.append({
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        })
        
        inputs = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            tokenize=True
        ).to(self.model.device)
        
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=self.config.do_sample,
                temperature=self.config.temperature if self.config.do_sample else None
            )
        
        response = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        return response.strip()


def load_model(device: str = "cuda") -> LFM2VL8B:
    """Load LFM2-VL-450M model."""
    return LFM2VL8B(device=device)
