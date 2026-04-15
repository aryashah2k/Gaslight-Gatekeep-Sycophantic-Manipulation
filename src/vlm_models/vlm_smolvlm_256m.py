"""
SmolVLM-256M Vision-Language Model Wrapper

HuggingFace: https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct

SmolVLM is a compact vision-language model using SigLIP for vision encoding.
Extremely efficient while maintaining reasonable performance.
"""

import torch
import numpy as np
from typing import List, Optional, Union
from PIL import Image
import logging

from .base_vlm import BaseVLM, VLMConfig

logger = logging.getLogger(__name__)


class SmolVLM256M(BaseVLM):
    """SmolVLM-256M-Instruct wrapper."""
    
    MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
    MODEL_NAME = "smolvlm_256m"
    HUGGINGFACE_URL = "https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct"
    
    def load_model(self):
        """Load SmolVLM model and processor."""
        from transformers import AutoProcessor, AutoModelForVision2Seq
        
        logger.info(f"Loading {self.MODEL_ID}...")
        
        self._processor = AutoProcessor.from_pretrained(
            self.MODEL_ID,
            trust_remote_code=True
        )
        
        self._model = AutoModelForVision2Seq.from_pretrained(
            self.MODEL_ID,
            torch_dtype=self.config.dtype,
            device_map="auto",
            trust_remote_code=True
        )
        
        self._model.eval()
        self._is_loaded = True
        
        logger.info(f"Loaded {self.MODEL_NAME} successfully")
    
    def get_vision_features(
        self,
        images: Union[Image.Image, List[Image.Image]]
    ) -> np.ndarray:
        """Extract SigLIP vision encoder features."""
        images = self._ensure_list(images)
        
        # Ensure model is loaded via property access
        model = self.model
        processor = self.processor
        
        all_features = []
        
        with torch.no_grad():
            for image in images:
                # Process image - SmolVLM requires <image> placeholder in text
                inputs = processor(
                    images=image,
                    text="<image>Describe this image.",
                    return_tensors="pt"
                ).to(self.device)
                
                # SmolVLM architecture: try different methods to get vision features
                pixel_values = inputs.get('pixel_values')
                
                if pixel_values is not None:
                    # Method 1: Direct vision_model access
                    if hasattr(model, 'vision_model') and model.vision_model is not None:
                        try:
                            vision_outputs = model.vision_model(pixel_values)
                            if hasattr(vision_outputs, 'last_hidden_state'):
                                features = vision_outputs.last_hidden_state.mean(dim=1)
                            else:
                                features = vision_outputs.mean(dim=1) if vision_outputs.dim() > 2 else vision_outputs
                            all_features.append(features.cpu().float().numpy())
                            continue
                        except Exception as e:
                            logger.debug(f"vision_model access failed: {e}")
                    
                    # Method 2: Try model.model.vision_model (nested structure)
                    if hasattr(model, 'model') and hasattr(model.model, 'vision_model'):
                        try:
                            vision_outputs = model.model.vision_model(pixel_values)
                            if hasattr(vision_outputs, 'last_hidden_state'):
                                features = vision_outputs.last_hidden_state.mean(dim=1)
                            else:
                                features = vision_outputs.mean(dim=1) if vision_outputs.dim() > 2 else vision_outputs
                            all_features.append(features.cpu().float().numpy())
                            continue
                        except Exception as e:
                            logger.debug(f"model.model.vision_model access failed: {e}")
                
                # Method 3: Use full model forward with hidden states
                try:
                    outputs = model(**inputs, output_hidden_states=True)
                    # Get hidden states and extract vision portion
                    if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                        # Use an early layer (more visual, less language)
                        hidden_states = outputs.hidden_states
                        # Take the last hidden state and pool
                        features = hidden_states[-1].mean(dim=1)
                    else:
                        # Use logits or last output
                        features = outputs.logits.mean(dim=1) if hasattr(outputs, 'logits') else outputs[0].mean(dim=1)
                    
                    all_features.append(features.cpu().float().numpy())
                except Exception as e:
                    logger.error(f"Failed to extract features: {e}")
                    raise
        
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
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        })
        
        # Apply chat template
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        
        # Process inputs
        inputs = self.processor(
            images=image,
            text=text,
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
        
        # Decode
        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
        response = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        
        return response.strip()


class SmolVLM500M(SmolVLM256M):
    """SmolVLM-500M-Instruct wrapper."""
    
    MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"
    MODEL_NAME = "smolvlm_500m"
    HUGGINGFACE_URL = "https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct"


def load_model(device: str = "cuda", size: str = "256m") -> SmolVLM256M:
    """Load SmolVLM model."""
    if size == "500m":
        return SmolVLM500M(device=device)
    return SmolVLM256M(device=device)
