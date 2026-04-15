"""
Qwen2-VL 2B Vision-Language Model Wrapper

HuggingFace: https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct

Qwen2-VL uses a custom vision encoder with Naive Dynamic Resolution.
The vision encoder is accessible via model.visual.

FIXED: Now extracts features from Qwen2-VL's visual encoder directly.
Note: Qwen2-VL's visual encoder requires grid_thw (grid thumbnail) info
which is computed by the processor. We handle this specially.
"""

import torch
import numpy as np
from typing import List, Optional, Union
from PIL import Image
import logging

from .base_vlm import BaseVLM, VLMConfig

logger = logging.getLogger(__name__)


class Qwen2VL2B(BaseVLM):
    """Qwen2-VL-2B-Instruct wrapper."""
    
    MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
    MODEL_NAME = "qwen2vl_2b"
    HUGGINGFACE_URL = "https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct"
    
    def load_model(self):
        """Load Qwen2-VL model and processor."""
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        
        logger.info(f"Loading {self.MODEL_ID}...")
        
        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.MODEL_ID,
            torch_dtype=self.config.dtype,
            device_map="auto",
            trust_remote_code=self.config.trust_remote_code
        )
        
        self._processor = AutoProcessor.from_pretrained(
            self.MODEL_ID,
            trust_remote_code=self.config.trust_remote_code
        )
        
        self._model.eval()
        self._is_loaded = True
        
        logger.info(f"Loaded {self.MODEL_NAME} successfully")
    
    def get_vision_features(
        self,
        images: Union[Image.Image, List[Image.Image]]
    ) -> np.ndarray:
        """Extract vision features from Qwen2-VL's visual encoder.
        
        Qwen2-VL uses Naive Dynamic Resolution which maps images to variable
        numbers of visual tokens. The visual encoder (model.visual) requires
        both pixel_values and grid_thw (grid thumbnail info).
        """
        images = self._ensure_list(images)
        
        all_features = []
        
        with torch.no_grad():
            for image in images:
                # Ensure RGB
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Prepare inputs with a dummy text
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": "Describe this image."}
                        ]
                    }
                ]
                
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                
                inputs = self.processor(
                    text=[text],
                    images=[image],
                    return_tensors="pt",
                    padding=True
                ).to(self.device)
                
                pixel_values = inputs.get('pixel_values')
                image_grid_thw = inputs.get('image_grid_thw')
                features = None
                
                # Try to access visual encoder directly
                if pixel_values is not None and hasattr(self.model, 'visual'):
                    try:
                        # Qwen2-VL's visual encoder needs grid_thw
                        if image_grid_thw is not None:
                            vision_outputs = self.model.visual(
                                pixel_values,
                                grid_thw=image_grid_thw
                            )
                        else:
                            vision_outputs = self.model.visual(pixel_values)
                        
                        # vision_outputs is typically (batch, num_tokens, hidden_dim)
                        if isinstance(vision_outputs, tuple):
                            vision_outputs = vision_outputs[0]
                        
                        # Mean pool over visual tokens
                        features = vision_outputs.mean(dim=1)
                    except Exception as e:
                        logger.debug(f"visual encoder access failed: {e}")
                
                # Fallback: Use full model hidden states
                if features is None:
                    outputs = self.model(**inputs, output_hidden_states=True)
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
        
        # Build conversation format
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
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True
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
        generated = self.processor.batch_decode(
            output_ids[:, inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )[0]
        
        return generated.strip()


def load_model(device: str = "cuda") -> Qwen2VL2B:
    """Load Qwen2-VL-2B model."""
    return Qwen2VL2B(device=device)
