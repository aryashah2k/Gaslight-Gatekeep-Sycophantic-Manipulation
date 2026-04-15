"""
BLIP-2 OPT 2.7B Vision-Language Model Wrapper

HuggingFace: https://huggingface.co/Salesforce/blip2-opt-2.7b

BLIP-2 is Salesforce's vision-language model that bridges frozen image encoders
and frozen LLMs using a lightweight Querying Transformer (Q-Former).

Replaces MiniCPM-V 2.6 in the pipeline.
"""

import torch
import numpy as np
from typing import List, Optional, Union
from PIL import Image
import logging

from .base_vlm import BaseVLM, VLMConfig

logger = logging.getLogger(__name__)


class BLIP2OPT27B(BaseVLM):
    """BLIP-2 OPT 2.7B vision-language model wrapper."""
    
    MODEL_ID = "Salesforce/blip2-opt-2.7b"
    MODEL_NAME = "blip2_opt27b"
    HUGGINGFACE_URL = "https://huggingface.co/Salesforce/blip2-opt-2.7b"
    
    def __init__(self, config: Optional[VLMConfig] = None, device: Optional[str] = None):
        super().__init__(config, device)
        self._vision_model = None  # Separate vision model for feature extraction
    
    @property
    def vision_model(self):
        """Lazy load vision model on first access."""
        if not self._is_loaded:
            self.load_model()
        return self._vision_model
    
    def load_model(self):
        """Load BLIP-2 model and processor."""
        from transformers import Blip2Processor, Blip2ForConditionalGeneration, Blip2Model
        
        logger.info(f"Loading {self.MODEL_ID}...")
        
        # Load processor
        self._processor = Blip2Processor.from_pretrained(self.MODEL_ID)
        
        # Load the base Blip2Model for feature extraction (has get_image_features method)
        self._vision_model = Blip2Model.from_pretrained(
            self.MODEL_ID,
            torch_dtype=torch.float16,
        ).eval().to(self.device)
        
        # Load the full model for generation
        self._model = Blip2ForConditionalGeneration.from_pretrained(
            self.MODEL_ID,
            torch_dtype=torch.float16,
        ).eval().to(self.device)
        
        self._is_loaded = True
        logger.info(f"Loaded {self.MODEL_NAME} successfully")
    
    def get_vision_features(
        self,
        images: Union[Image.Image, List[Image.Image]]
    ) -> np.ndarray:
        """Extract vision features from BLIP-2.
        
        Uses Blip2Model.get_image_features() which returns vision encoder outputs.
        """
        images = self._ensure_list(images)
        
        all_features = []
        
        with torch.no_grad():
            for image in images:
                # Ensure RGB
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                try:
                    # Process image only (no text needed for feature extraction)
                    # Access processor via property to trigger lazy loading
                    inputs = self.processor(
                        images=image,
                        return_tensors="pt"
                    ).to(device=self.device, dtype=torch.float16)
                    
                    # Use get_image_features from Blip2Model (via property)
                    # This returns BaseModelOutputWithPooling with pooler_output
                    vision_outputs = self.vision_model.get_image_features(
                        pixel_values=inputs.pixel_values
                    )
                    
                    # Get pooler_output (batch_size, hidden_size)
                    if hasattr(vision_outputs, 'pooler_output') and vision_outputs.pooler_output is not None:
                        features = vision_outputs.pooler_output
                    else:
                        # Fallback to last_hidden_state mean
                        features = vision_outputs.last_hidden_state.mean(dim=1)
                    
                except Exception as e:
                    logger.warning(f"Feature extraction failed: {e}")
                    # ViT hidden size for BLIP-2 is 1408
                    features = torch.zeros(1, 1408, device=self.device, dtype=torch.float16)
                
                all_features.append(features.cpu().float().numpy())
        
        return np.concatenate(all_features, axis=0)
    
    def generate(
        self,
        image: Image.Image,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate response using BLIP-2's generate method."""
        max_tokens = max_tokens or self.config.max_new_tokens
        
        # Ensure RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        try:
            # Process inputs as per official example (access via property)
            inputs = self.processor(
                images=image,
                text=prompt,
                return_tensors="pt"
            ).to(device=self.device, dtype=torch.float16)
            
            # Generate using the full model (access via property)
            with torch.inference_mode():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False
                )
                
                # Decode
                decoded = self.processor.decode(output[0], skip_special_tokens=True)
            
            return decoded.strip()
                
        except Exception as e:
            logger.error(f"Generation failed for {self.MODEL_NAME}: {e}")
            return "UNCLEAR"
    
    def unload(self):
        """Unload models from memory."""
        super().unload()
        
        if self._vision_model is not None:
            del self._vision_model
            self._vision_model = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_model(device: str = "cuda") -> BLIP2OPT27B:
    """Load BLIP-2 OPT 2.7B model."""
    return BLIP2OPT27B(device=device)
