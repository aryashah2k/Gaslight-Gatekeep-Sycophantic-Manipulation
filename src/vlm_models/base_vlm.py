"""
Base VLM Abstract Class

This module defines the abstract base class for all VLM wrappers.
Each VLM implementation inherits from this class and implements
the required methods for feature extraction and text generation.
"""

import torch
import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from PIL import Image
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class VLMConfig:
    """Configuration for VLM loading and inference."""
    device: str = "cuda"
    dtype: torch.dtype = torch.float16
    max_new_tokens: int = 256
    temperature: float = 0.0  # Greedy for reproducibility
    do_sample: bool = False
    trust_remote_code: bool = True


class BaseVLM(ABC):
    """
    Abstract base class for Vision-Language Model wrappers.
    
    Each VLM wrapper must implement:
    - load_model(): Load the model and processor
    - get_vision_features(): Extract frozen vision encoder features
    - generate(): Generate text response given image and prompt
    
    Example:
        >>> vlm = ConcreteVLM(device="cuda")
        >>> features = vlm.get_vision_features(image)
        >>> response = vlm.generate(image, "What is in this image?")
    """
    
    # Class attributes - override in subclasses
    MODEL_ID: str = ""  # HuggingFace model ID
    MODEL_NAME: str = ""  # Short name for files
    HUGGINGFACE_URL: str = ""  # Reference URL
    
    def __init__(
        self,
        config: Optional[VLMConfig] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the VLM wrapper.
        
        Args:
            config: VLM configuration
            device: Override device (e.g., "cuda", "cuda:0", "cpu")
        """
        self.config = config or VLMConfig()
        
        if device:
            self.config.device = device
        
        self.device = torch.device(self.config.device)
        
        # Model and processor - loaded lazily
        self._model = None
        self._processor = None
        self._is_loaded = False
        
        logger.info(f"Initialized {self.MODEL_NAME} wrapper (device={self.device})")
    
    @property
    def model(self):
        """Lazy load model on first access."""
        if not self._is_loaded:
            self.load_model()
        return self._model
    
    @property
    def processor(self):
        """Lazy load processor on first access."""
        if not self._is_loaded:
            self.load_model()
        return self._processor
    
    @abstractmethod
    def load_model(self):
        """
        Load the model and processor from HuggingFace.
        
        Must set self._model and self._processor.
        Should handle device placement and dtype.
        """
        pass
    
    @abstractmethod
    def get_vision_features(
        self,
        image: Union[Image.Image, List[Image.Image]]
    ) -> np.ndarray:
        """
        Extract frozen vision encoder features.
        
        Args:
            image: Single PIL Image or list of images
            
        Returns:
            Feature array of shape (n_images, feature_dim)
        """
        pass
    
    @abstractmethod
    def generate(
        self,
        image: Image.Image,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate text response given image and prompt.
        
        Args:
            image: PIL Image
            prompt: User prompt/question
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text response
        """
        pass
    
    def _ensure_list(
        self, 
        images: Union[Image.Image, List[Image.Image]]
    ) -> List[Image.Image]:
        """Ensure images is a list."""
        if isinstance(images, Image.Image):
            return [images]
        return images
    
    def unload(self):
        """Unload model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
        
        if self._processor is not None:
            del self._processor
            self._processor = None
        
        self._is_loaded = False
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info(f"Unloaded {self.MODEL_NAME}")
    
    def get_feature_extractor(self):
        """
        Get a FeatureExtractor instance for this model.
        
        Returns:
            Configured FeatureExtractor
        """
        from src.stage1_brain_score.feature_extractor import FeatureExtractor, ExtractionConfig
        
        # Create a custom extraction wrapper
        class VLMFeatureExtractor(FeatureExtractor):
            def __init__(self, vlm: BaseVLM):
                self.vlm = vlm
                self.model_name = vlm.MODEL_NAME
                self.config = ExtractionConfig(device=str(vlm.device))
                self._feature_dim = None
            
            def extract(self, image_paths, batch_size=16, show_progress=True):
                from tqdm import tqdm
                from src.stage1_brain_score.feature_extractor import ExtractedFeatures
                
                all_features = []
                
                iterator = range(0, len(image_paths), batch_size)
                if show_progress:
                    iterator = tqdm(iterator, desc=f"Extracting {self.model_name}")
                
                for start_idx in iterator:
                    end_idx = min(start_idx + batch_size, len(image_paths))
                    batch_paths = image_paths[start_idx:end_idx]
                    
                    images = [Image.open(p).convert("RGB") for p in batch_paths]
                    features = self.vlm.get_vision_features(images)
                    all_features.append(features)
                
                features_array = np.concatenate(all_features, axis=0)
                
                # L2 normalize
                norms = np.linalg.norm(features_array, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-8)
                features_array = features_array / norms
                
                self._feature_dim = features_array.shape[1]
                
                return ExtractedFeatures(
                    features=features_array,
                    model_name=self.model_name,
                    feature_dim=self._feature_dim,
                    n_images=len(image_paths),
                    pooling_method="model_specific",
                    extraction_config={"batch_size": batch_size}
                )
        
        return VLMFeatureExtractor(self)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_id='{self.MODEL_ID}', device='{self.device}')"
