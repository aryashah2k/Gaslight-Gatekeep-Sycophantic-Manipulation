"""
Gemma-3 1B Vision-Language Model Wrapper

HuggingFace: https://huggingface.co/google/gemma-3-4b-it

Gemma-3 uses SigLIP as its vision encoder. We access vision_tower directly
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


class Gemma3VLM(BaseVLM):
    """Gemma-3 4B Vision wrapper (named gemma3_1b for compatibility)."""
    
    MODEL_ID = "google/gemma-3-4b-it"
    MODEL_NAME = "gemma3_1b"
    HUGGINGFACE_URL = "https://huggingface.co/google/gemma-3-4b-it"
    
    def load_model(self):
        """Load Gemma-3 model and processor."""
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration
        
        logger.info(f"Loading {self.MODEL_ID}...")
        
        self._processor = AutoProcessor.from_pretrained(self.MODEL_ID)
        
        # Gemma-3 needs float32 for stability
        self._model = Gemma3ForConditionalGeneration.from_pretrained(
            self.MODEL_ID,
            torch_dtype=torch.float32,
            device_map="auto"
        ).eval()
        
        self._is_loaded = True
        
        logger.info(f"Loaded {self.MODEL_NAME} successfully (float32 mode)")
    
    def get_vision_features(
        self,
        images: Union[Image.Image, List[Image.Image]]
    ) -> np.ndarray:
        """Extract vision features from SigLIP vision_tower directly.
        
        Gemma-3 uses SigLIP as vision encoder accessible via model.vision_tower.
        """
        images = self._ensure_list(images)
        
        all_features = []
        
        with torch.no_grad():
            for image in images:
                # Ensure RGB
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Process image
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": "Describe this image."}
                        ]
                    }
                ]
                
                inputs = self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt"
                )
                
                # Move inputs to device
                inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v 
                          for k, v in inputs.items()}
                
                pixel_values = inputs.get('pixel_values')
                features = None
                
                # Try to access vision_tower directly
                if pixel_values is not None:
                    # Method 1: model.vision_tower (Gemma-3 standard path)
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
                
                # Fallback: Use full model hidden states
                if features is None:
                    outputs = self.model(**inputs, output_hidden_states=True, use_cache=False)
                    last_hidden = outputs.hidden_states[-1]
                    features = last_hidden.mean(dim=1)
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
        
        # Build conversation
        messages = []
        
        if system_prompt:
            messages.append({
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            })
        
        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        })
        
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v 
                  for k, v in inputs.items()}
        
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


def load_model(device: str = "cuda") -> Gemma3VLM:
    """Load Gemma-3 VLM model."""
    return Gemma3VLM(device=device)
