"""
LLaVA-1.6-7B Vision-Language Model Wrapper

HuggingFace: https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf

LLaVA (Large Language and Vision Assistant) uses a CLIP vision encoder
combined with Mistral-7B language model.

FIXED: Now extracts features from vision_tower (CLIP ViT) directly
instead of LLM hidden states.
"""

import torch
import numpy as np
from typing import List, Optional, Union
from PIL import Image
import logging

from .base_vlm import BaseVLM, VLMConfig

logger = logging.getLogger(__name__)


class Llava7B(BaseVLM):
    """LLaVA-1.6-Mistral-7B wrapper."""
    
    MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"
    MODEL_NAME = "llava_7b"
    HUGGINGFACE_URL = "https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf"
    
    def load_model(self):
        """Load LLaVA model and processor."""
        from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
        
        logger.info(f"Loading {self.MODEL_ID}...")
        
        self._processor = LlavaNextProcessor.from_pretrained(self.MODEL_ID)
        
        self._model = LlavaNextForConditionalGeneration.from_pretrained(
            self.MODEL_ID,
            torch_dtype=self.config.dtype,
            device_map="auto"
        )
        
        self._model.eval()
        self._is_loaded = True
        
        logger.info(f"Loaded {self.MODEL_NAME} successfully")
    
    def get_vision_features(
        self,
        images: Union[Image.Image, List[Image.Image]]
    ) -> np.ndarray:
        """Extract vision features directly from CLIP vision tower.
        
        LLaVA uses CLIP ViT as vision encoder. We access it via model.vision_tower
        to get pure visual representations without LLM contamination.
        """
        images = self._ensure_list(images)
        
        all_features = []
        
        with torch.no_grad():
            for image in images:
                # Ensure RGB
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Process image to get pixel_values
                inputs = self.processor(
                    images=image,
                    text="<image>\nDescribe this image.",
                    return_tensors="pt"
                ).to(self.device)
                
                pixel_values = inputs.get('pixel_values')
                
                if pixel_values is not None and hasattr(self.model, 'vision_tower'):
                    # Access the CLIP vision tower directly
                    vision_tower = self.model.vision_tower
                    
                    # LLaVA-Next may have image_sizes for dynamic resolution
                    image_sizes = inputs.get('image_sizes')
                    
                    # Handle different pixel_values shapes
                    # LLaVA-Next uses (batch, num_patches, channels, height, width)
                    if pixel_values.dim() == 5:
                        # Flatten patches for vision tower
                        batch_size, num_patches, c, h, w = pixel_values.shape
                        pixel_values_flat = pixel_values.view(batch_size * num_patches, c, h, w)
                    else:
                        pixel_values_flat = pixel_values
                    
                    # Get vision tower output
                    vision_outputs = vision_tower(
                        pixel_values_flat,
                        output_hidden_states=True
                    )
                    
                    # Use last hidden state and pool
                    if hasattr(vision_outputs, 'last_hidden_state'):
                        hidden_state = vision_outputs.last_hidden_state
                    elif hasattr(vision_outputs, 'hidden_states'):
                        hidden_state = vision_outputs.hidden_states[-1]
                    else:
                        # Fallback - use the output directly
                        hidden_state = vision_outputs[0] if isinstance(vision_outputs, tuple) else vision_outputs
                    
                    # Mean pool over spatial dimensions
                    # Shape: (batch, seq_len, hidden_dim) -> (batch, hidden_dim)
                    features = hidden_state.mean(dim=1)
                    
                    # If we had multiple patches, mean pool across patches too
                    if pixel_values.dim() == 5:
                        features = features.view(batch_size, num_patches, -1).mean(dim=1)
                else:
                    # Fallback to full model hidden states (not recommended)
                    logger.warning("vision_tower not accessible, using LLM hidden states")
                    outputs = self.model(
                        **inputs,
                        output_hidden_states=True,
                        use_cache=False
                    )
                    last_hidden = outputs.hidden_states[-1]
                    features = last_hidden.mean(dim=1)
                
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
        
        # Build conversation format
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n<image>\n{prompt}"
        else:
            full_prompt = f"<image>\n{prompt}"
        
        inputs = self.processor(
            text=full_prompt,
            images=image,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=self.config.do_sample,
                temperature=self.config.temperature if self.config.do_sample else None
            )
        
        # Decode, skipping input tokens
        response = self.processor.decode(
            output_ids[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return response.strip()


def load_model(device: str = "cuda") -> Llava7B:
    """Load LLaVA-7B model."""
    return Llava7B(device=device)
