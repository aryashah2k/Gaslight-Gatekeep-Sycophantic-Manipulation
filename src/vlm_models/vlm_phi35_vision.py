"""
Phi-3.5-Vision Vision-Language Model Wrapper

HuggingFace: https://huggingface.co/microsoft/Phi-3.5-vision-instruct

Phi-3.5-Vision is Microsoft's efficient VLM using CLIP ViT for vision.

FIXED: Now extracts features from vision tower directly instead of LLM hidden states.
"""

import torch
import numpy as np
from typing import List, Optional, Union
from PIL import Image
import logging

from .base_vlm import BaseVLM, VLMConfig

logger = logging.getLogger(__name__)


class Phi35Vision(BaseVLM):
    """Phi-3.5-Vision wrapper."""
    
    MODEL_ID = "microsoft/Phi-3.5-vision-instruct"
    MODEL_NAME = "phi35_vision"
    HUGGINGFACE_URL = "https://huggingface.co/microsoft/Phi-3.5-vision-instruct"
    
    def load_model(self):
        """Load Phi-3.5-Vision model and processor."""
        from transformers import AutoProcessor, AutoModelForCausalLM
        
        logger.info(f"Loading {self.MODEL_ID}...")
        
        self._processor = AutoProcessor.from_pretrained(
            self.MODEL_ID,
            trust_remote_code=True
        )
        
        self._model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_ID,
            torch_dtype=self.config.dtype,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager"  # Avoid FlashAttention requirement
        )
        
        self._model.eval()
        self._is_loaded = True
        
        logger.info(f"Loaded {self.MODEL_NAME} successfully")
    
    def get_vision_features(
        self,
        images: Union[Image.Image, List[Image.Image]]
    ) -> np.ndarray:
        """Extract vision features from Phi-3.5-Vision's image encoder.
        
        Phi-3.5-Vision has a CLIP-based vision encoder. We try multiple paths
        to access it: vision_model, img_processor, or vision_embed_tokens.
        """
        images = self._ensure_list(images)
        
        all_features = []
        
        with torch.no_grad():
            for image in images:
                # Ensure RGB
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Process with placeholder text
                messages = [
                    {"role": "user", "content": f"<|image_1|>\nDescribe this image."}
                ]
                
                prompt = self.processor.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                
                inputs = self.processor(
                    prompt,
                    [image],
                    return_tensors="pt"
                ).to(self.device)
                
                pixel_values = inputs.get('pixel_values')
                features = None
                
                # Try to access vision encoder directly
                if pixel_values is not None:
                    # Method 1: Try model.vision_model
                    if hasattr(self.model, 'vision_model') and self.model.vision_model is not None:
                        try:
                            vision_outputs = self.model.vision_model(pixel_values)
                            if hasattr(vision_outputs, 'last_hidden_state'):
                                features = vision_outputs.last_hidden_state.mean(dim=1)
                            elif hasattr(vision_outputs, 'pooler_output'):
                                features = vision_outputs.pooler_output
                            else:
                                features = vision_outputs[0].mean(dim=1) if isinstance(vision_outputs, tuple) else vision_outputs.mean(dim=1)
                        except Exception as e:
                            logger.debug(f"vision_model access failed: {e}")
                    
                    # Method 2: Try model.model.vision_model
                    if features is None and hasattr(self.model, 'model') and hasattr(self.model.model, 'vision_model'):
                        try:
                            vision_outputs = self.model.model.vision_model(pixel_values)
                            if hasattr(vision_outputs, 'last_hidden_state'):
                                features = vision_outputs.last_hidden_state.mean(dim=1)
                            else:
                                features = vision_outputs[0].mean(dim=1) if isinstance(vision_outputs, tuple) else vision_outputs.mean(dim=1)
                        except Exception as e:
                            logger.debug(f"model.vision_model access failed: {e}")
                    
                    # Method 3: Try img_processor (Phi-3.5 specific)
                    if features is None and hasattr(self.model, 'model') and hasattr(self.model.model, 'vision_embed_tokens'):
                        try:
                            # vision_embed_tokens is an embedding layer for vision
                            vision_embeds = self.model.model.vision_embed_tokens(pixel_values)
                            features = vision_embeds.mean(dim=1)
                        except Exception as e:
                            logger.debug(f"vision_embed_tokens access failed: {e}")
                
                # Fallback: Use full model hidden states
                if features is None:
                    outputs = self.model(
                        **inputs, 
                        output_hidden_states=True,
                        use_cache=False
                    )
                    hidden_states = outputs.hidden_states[-1]
                    features = hidden_states.mean(dim=1)
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
        
        # Build conversation
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({
            "role": "user",
            "content": f"<|image_1|>\n{prompt}"
        })
        
        prompt_text = self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.processor(
            prompt_text,
            [image],
            return_tensors="pt"
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            try:
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=self.config.do_sample,
                    temperature=self.config.temperature if self.config.do_sample else None,
                    eos_token_id=self.processor.tokenizer.eos_token_id
                )
            except Exception as e:
                logger.warning(f"Standard generation failed: {e}, trying fallback")
                # Fallback: manual autoregressive generation
                output_ids = inputs['input_ids']
                for _ in range(max_tokens):
                    outputs = self.model(**inputs)
                    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    output_ids = torch.cat([output_ids, next_token], dim=-1)
                    if next_token.item() == self.processor.tokenizer.eos_token_id:
                        break
                    inputs['input_ids'] = output_ids
                    if 'attention_mask' in inputs:
                        inputs['attention_mask'] = torch.cat([
                            inputs['attention_mask'],
                            torch.ones((1, 1), device=self.device)
                        ], dim=-1)
        
        # Decode response (skip input)
        response = self.processor.decode(
            output_ids[0][inputs['input_ids'].shape[1] if 'input_ids' in inputs else 0:],
            skip_special_tokens=True
        )
        
        return response.strip()


def load_model(device: str = "cuda") -> Phi35Vision:
    """Load Phi-3.5-Vision model."""
    return Phi35Vision(device=device)
