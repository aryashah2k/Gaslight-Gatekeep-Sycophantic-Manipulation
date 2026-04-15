"""
Idefics2-8B Vision-Language Model Wrapper

HuggingFace: https://huggingface.co/HuggingFaceM4/idefics2-8b

Idefics2 uses a modified SigLIP encoder (Idefics2VisionTransformer).
Vision encoder hidden size is 1152.

CORRECT IMPLEMENTATION: Converts pixel_values to model dtype before vision extraction.
"""

import torch
import numpy as np
from typing import List, Optional, Union
from PIL import Image
import logging

from .base_vlm import BaseVLM, VLMConfig

logger = logging.getLogger(__name__)


class Idefics28B(BaseVLM):
    """Idefics2-8B wrapper."""
    
    MODEL_ID = "HuggingFaceM4/idefics2-8b"
    MODEL_NAME = "idefics2_8b"
    HUGGINGFACE_URL = "https://huggingface.co/HuggingFaceM4/idefics2-8b"
    VISION_HIDDEN_SIZE = 1152  # Idefics2VisionTransformer
    
    def load_model(self):
        """Load Idefics2 model and processor."""
        from transformers import AutoProcessor, AutoModelForVision2Seq
        
        logger.info(f"Loading {self.MODEL_ID}...")
        
        self._processor = AutoProcessor.from_pretrained(self.MODEL_ID)
        
        self._model = AutoModelForVision2Seq.from_pretrained(
            self.MODEL_ID,
            torch_dtype=self.config.dtype,
            device_map="auto"
        )
        
        self._model.eval()
        self._is_loaded = True
        
        # Validate vision encoder access
        if not hasattr(self.model, 'model'):
            raise RuntimeError(f"{self.MODEL_NAME}: model.model not found")
        
        inner_model = self.model.model
        if not hasattr(inner_model, 'vision_model'):
            raise RuntimeError(f"{self.MODEL_NAME}: model.model.vision_model not found")
        
        logger.info(f"Loaded {self.MODEL_NAME} successfully (vision_model: {type(inner_model.vision_model).__name__})")
    
    def get_vision_features(
        self,
        images: Union[Image.Image, List[Image.Image]]
    ) -> np.ndarray:
        """Extract vision features from Idefics2VisionTransformer.
        
        Converts pixel_values to model dtype and passes to vision_model.
        """
        images = self._ensure_list(images)
        all_features = []
        
        # Get model dtype for conversion
        model_dtype = next(self.model.parameters()).dtype
        
        with torch.no_grad():
            for i, image in enumerate(images):
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Process using documented API
                inputs = self.processor(
                    text="<image>",
                    images=[image],
                    return_tensors="pt"
                ).to(self.device)
                
                pixel_values = inputs.get('pixel_values')
                if pixel_values is None:
                    raise RuntimeError(f"{self.MODEL_NAME}: processor did not return pixel_values")
                
                # Convert pixel_values to match model dtype (float16)
                pixel_values = pixel_values.to(dtype=model_dtype)
                
                # Handle potential 5D shape: (batch, num_images, channels, h, w)
                if pixel_values.dim() == 5:
                    batch_size, num_images, c, h, w = pixel_values.shape
                    pixel_values = pixel_values.view(batch_size * num_images, c, h, w)
                
                # Access vision_model through model.model
                vision_model = self.model.model.vision_model
                
                try:
                    vision_outputs = vision_model(pixel_values)
                    
                    if hasattr(vision_outputs, 'last_hidden_state'):
                        # Mean pool over spatial tokens
                        features = vision_outputs.last_hidden_state.mean(dim=1)
                    elif hasattr(vision_outputs, 'pooler_output') and vision_outputs.pooler_output is not None:
                        features = vision_outputs.pooler_output
                    else:
                        hidden = vision_outputs[0] if isinstance(vision_outputs, tuple) else vision_outputs
                        features = hidden.mean(dim=1) if hidden.dim() > 2 else hidden
                    
                    # If we had multiple images, take the first one
                    if features.shape[0] > 1:
                        features = features[:1]
                    
                    logger.debug(f"Image {i}: Extracted via vision_model, shape {features.shape}")
                    
                except Exception as e:
                    raise RuntimeError(f"{self.MODEL_NAME}: vision_model extraction failed: {e}")
                
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
        
        # Build messages
        messages = []
        
        if system_prompt:
            messages.append({
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            })
        
        messages.append({
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]
        })
        
        # Apply chat template
        text_prompt = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        
        # Process inputs
        inputs = self.processor(
            text=text_prompt,
            images=[image],
            return_tensors="pt"
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=self.config.do_sample,
                temperature=self.config.temperature if self.config.do_sample else None
            )
        
        # Decode response
        response = self.processor.batch_decode(
            output_ids, skip_special_tokens=True
        )[0]
        
        # Extract assistant response
        if "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()
        
        return response.strip()


def load_model(device: str = "cuda") -> Idefics28B:
    """Load Idefics2-8B model."""
    return Idefics28B(device=device)
