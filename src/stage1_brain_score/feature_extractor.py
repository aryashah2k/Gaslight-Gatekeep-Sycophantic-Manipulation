"""
Feature Extraction Module for VLM Vision Encoders

This module provides a unified interface for extracting visual features
from frozen vision encoders of various VLMs.

The extracted features are used to train Ridge regression models that
predict fMRI responses, yielding the "Brain-Score" metric.
"""

import os
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass
from PIL import Image
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExtractionConfig:
    """Configuration for feature extraction."""
    batch_size: int = 16
    device: str = "cuda"
    dtype: torch.dtype = torch.float16
    pooling: str = "mean"  # 'mean', 'cls', 'last', 'flatten'
    normalize: bool = True
    save_intermediate: bool = True
    output_dir: Optional[str] = None


@dataclass 
class ExtractedFeatures:
    """Container for extracted features."""
    features: np.ndarray  # Shape: (n_images, feature_dim)
    model_name: str
    feature_dim: int
    n_images: int
    pooling_method: str
    extraction_config: Dict


class FeatureExtractor:
    """
    Unified feature extractor for VLM vision encoders.
    
    This class provides a common interface for extracting features from
    different VLM architectures. Subclasses implement model-specific logic.
    
    Features are extracted from the frozen vision encoder with no gradients,
    making this computationally efficient.
    
    Example:
        >>> extractor = FeatureExtractor(model_name="llava_7b", device="cuda")
        >>> features = extractor.extract(image_paths, batch_size=16)
        >>> print(features.shape)  # (n_images, feature_dim)
    """
    
    def __init__(
        self,
        model_name: str,
        model: torch.nn.Module,
        processor: Callable,
        config: Optional[ExtractionConfig] = None
    ):
        """
        Initialize the feature extractor.
        
        Args:
            model_name: Name identifier for the model
            model: The VLM model (will use vision encoder)
            processor: Preprocessing function/processor
            config: Extraction configuration
        """
        self.model_name = model_name
        self.model = model
        self.processor = processor
        self.config = config or ExtractionConfig()
        
        # Set model to eval mode and disable gradients
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        
        self._feature_dim = None
    
    @property
    def feature_dim(self) -> int:
        """Get the feature dimension (computed on first extraction)."""
        if self._feature_dim is None:
            raise ValueError("Feature dim not yet computed. Run extract() first.")
        return self._feature_dim
    
    def _preprocess_images(
        self, 
        images: List[Image.Image]
    ) -> torch.Tensor:
        """
        Preprocess images using the model's processor.
        
        Args:
            images: List of PIL Images
            
        Returns:
            Preprocessed tensor ready for model
        """
        # Default preprocessing - subclasses may override
        inputs = self.processor(
            images=images,
            return_tensors="pt",
            padding=True
        )
        return inputs
    
    def _extract_vision_features(
        self, 
        inputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Extract features from vision encoder.
        
        This is a default implementation. Subclasses should override
        for model-specific feature extraction.
        
        Args:
            inputs: Preprocessed model inputs
            
        Returns:
            Vision features tensor
        """
        # Move to device
        inputs = {k: v.to(self.config.device) for k, v in inputs.items() 
                  if isinstance(v, torch.Tensor)}
        
        with torch.no_grad():
            # Try common attribute names for vision encoder
            if hasattr(self.model, 'vision_tower'):
                outputs = self.model.vision_tower(inputs.get('pixel_values'))
            elif hasattr(self.model, 'visual'):
                outputs = self.model.visual(inputs.get('pixel_values'))
            elif hasattr(self.model, 'vision_model'):
                outputs = self.model.vision_model(inputs.get('pixel_values'))
            elif hasattr(self.model, 'get_vision_tower'):
                vision_tower = self.model.get_vision_tower()
                outputs = vision_tower(inputs.get('pixel_values'))
            else:
                # Fallback: use forward hook to capture features
                raise NotImplementedError(
                    f"Vision encoder extraction not implemented for {self.model_name}. "
                    "Please use a model-specific extractor."
                )
        
        return outputs
    
    def _pool_features(
        self, 
        features: torch.Tensor,
        pooling: str = "mean"
    ) -> torch.Tensor:
        """
        Pool spatial/sequence features to a single vector.
        
        Args:
            features: Feature tensor of shape (B, S, D) or (B, H, W, D)
            pooling: Pooling method ('mean', 'cls', 'last', 'flatten')
            
        Returns:
            Pooled features of shape (B, D) or (B, D*S) for flatten
        """
        if len(features.shape) == 2:
            # Already 2D, no pooling needed
            return features
        
        if pooling == "mean":
            # Mean over spatial/sequence dimension
            if len(features.shape) == 3:
                return features.mean(dim=1)
            elif len(features.shape) == 4:
                return features.mean(dim=(1, 2))
        
        elif pooling == "cls":
            # Use first token (CLS token for ViT)
            return features[:, 0]
        
        elif pooling == "last":
            # Use last token
            return features[:, -1]
        
        elif pooling == "flatten":
            # Flatten all spatial dims
            return features.reshape(features.shape[0], -1)
        
        else:
            raise ValueError(f"Unknown pooling method: {pooling}")
    
    def _normalize_features(
        self, 
        features: np.ndarray
    ) -> np.ndarray:
        """L2 normalize features."""
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)  # Avoid division by zero
        return features / norms
    
    def extract(
        self,
        image_paths: List[Path],
        batch_size: Optional[int] = None,
        show_progress: bool = True
    ) -> ExtractedFeatures:
        """
        Extract features from a list of images.
        
        Args:
            image_paths: List of paths to images
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
            
        Returns:
            ExtractedFeatures object containing the extracted features
        """
        batch_size = batch_size or self.config.batch_size
        all_features = []
        
        # Process in batches
        iterator = range(0, len(image_paths), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc=f"Extracting {self.model_name}")
        
        for start_idx in iterator:
            end_idx = min(start_idx + batch_size, len(image_paths))
            batch_paths = image_paths[start_idx:end_idx]
            
            # Load images
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            
            # Preprocess
            inputs = self._preprocess_images(images)
            
            # Extract features
            features = self._extract_vision_features(inputs)
            
            # Handle different output formats
            if isinstance(features, tuple):
                features = features[0]
            if hasattr(features, 'last_hidden_state'):
                features = features.last_hidden_state
            
            # Pool features
            pooled = self._pool_features(features, self.config.pooling)
            
            # Convert to numpy
            batch_features = pooled.cpu().float().numpy()
            all_features.append(batch_features)
        
        # Concatenate all batches
        features_array = np.concatenate(all_features, axis=0)
        
        # Normalize if requested
        if self.config.normalize:
            features_array = self._normalize_features(features_array)
        
        # Store feature dimension
        self._feature_dim = features_array.shape[1]
        
        return ExtractedFeatures(
            features=features_array,
            model_name=self.model_name,
            feature_dim=self._feature_dim,
            n_images=len(image_paths),
            pooling_method=self.config.pooling,
            extraction_config={
                "batch_size": batch_size,
                "normalize": self.config.normalize,
                "pooling": self.config.pooling
            }
        )
    
    def save_features(
        self,
        features: ExtractedFeatures,
        output_dir: Union[str, Path],
        subject_id: Optional[int] = None
    ) -> Path:
        """
        Save extracted features to disk.
        
        Args:
            features: ExtractedFeatures object
            output_dir: Output directory
            subject_id: Optional subject ID for filename
            
        Returns:
            Path to saved file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename
        if subject_id:
            filename = f"{features.model_name}_subj{subject_id:02d}_features.npz"
        else:
            filename = f"{features.model_name}_features.npz"
        
        output_path = output_dir / filename
        
        # Save as compressed npz
        np.savez_compressed(
            output_path,
            features=features.features,
            model_name=features.model_name,
            feature_dim=features.feature_dim,
            n_images=features.n_images,
            pooling_method=features.pooling_method
        )
        
        logger.info(f"Saved features to {output_path}")
        return output_path
    
    @staticmethod
    def load_features(path: Union[str, Path]) -> ExtractedFeatures:
        """
        Load features from disk.
        
        Args:
            path: Path to saved features file
            
        Returns:
            ExtractedFeatures object
        """
        data = np.load(path, allow_pickle=True)
        
        return ExtractedFeatures(
            features=data['features'],
            model_name=str(data['model_name']),
            feature_dim=int(data['feature_dim']),
            n_images=int(data['n_images']),
            pooling_method=str(data['pooling_method']),
            extraction_config={}
        )


def create_extractor_for_model(
    model_name: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16
) -> FeatureExtractor:
    """
    Factory function to create appropriate extractor for a model.
    
    This function dynamically imports the model-specific module and
    creates the appropriate extractor.
    
    Args:
        model_name: Name of the model (e.g., 'qwen2vl_2b', 'llava_7b')
        device: Device to use
        dtype: Data type for model
        
    Returns:
        Configured FeatureExtractor instance
    """
    # Import model-specific module
    module_name = f"src.vlm_models.vlm_{model_name}"
    
    try:
        import importlib
        vlm_module = importlib.import_module(module_name)
        
        # Get the VLM class (convention: class name is uppercase version)
        class_name = ''.join(word.title() for word in model_name.split('_'))
        vlm_class = getattr(vlm_module, class_name)
        
        # Create VLM instance
        vlm = vlm_class(device=device)
        
        # Return its extractor
        return vlm.get_feature_extractor()
        
    except (ImportError, AttributeError) as e:
        raise ValueError(f"Could not create extractor for {model_name}: {e}")
