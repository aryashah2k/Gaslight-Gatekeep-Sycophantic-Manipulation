"""
COCO Annotation Utilities for Algonauts Images

This module provides utilities for parsing COCO annotations and extracting
object/scene information for gaslighting prompt generation.

The Algonauts images are cropped versions of COCO images. The NSD ID in the
filename can be used to map back to the original COCO image ID.

References:
    - COCO Dataset: https://cocodataset.org/
    - NSD: https://naturalscenesdataset.org/
"""

import os
import json
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ImageAnnotation:
    """Container for image annotation data."""
    image_id: str
    image_path: Path
    nsd_id: int
    objects: List[str]
    object_counts: Dict[str, int]
    caption: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    relationships: List[str] = field(default_factory=list)


# COCO category ID to name mapping (80 categories)
COCO_CATEGORIES = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane",
    6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic light",
    11: "fire hydrant", 13: "stop sign", 14: "parking meter", 15: "bench",
    16: "bird", 17: "cat", 18: "dog", 19: "horse", 20: "sheep",
    21: "cow", 22: "elephant", 23: "bear", 24: "zebra", 25: "giraffe",
    27: "backpack", 28: "umbrella", 31: "handbag", 32: "tie", 33: "suitcase",
    34: "frisbee", 35: "skis", 36: "snowboard", 37: "sports ball", 38: "kite",
    39: "baseball bat", 40: "baseball glove", 41: "skateboard", 42: "surfboard",
    43: "tennis racket", 44: "bottle", 46: "wine glass", 47: "cup", 48: "fork",
    49: "knife", 50: "spoon", 51: "bowl", 52: "banana", 53: "apple",
    54: "sandwich", 55: "orange", 56: "broccoli", 57: "carrot", 58: "hot dog",
    59: "pizza", 60: "donut", 61: "cake", 62: "chair", 63: "couch",
    64: "potted plant", 65: "bed", 67: "dining table", 70: "toilet", 72: "tv",
    73: "laptop", 74: "mouse", 75: "remote", 76: "keyboard", 77: "cell phone",
    78: "microwave", 79: "oven", 80: "toaster", 81: "sink", 82: "refrigerator",
    84: "book", 85: "clock", 86: "vase", 87: "scissors", 88: "teddy bear",
    89: "hair drier", 90: "toothbrush"
}

# Confusable object pairs for gaslighting (visually similar objects)
CONFUSABLE_OBJECTS = {
    "bus": ["truck", "van", "train"],
    "truck": ["bus", "van", "car"],
    "car": ["truck", "van", "taxi"],
    "dog": ["cat", "wolf", "fox"],
    "cat": ["dog", "rabbit", "fox"],
    "bicycle": ["motorcycle", "scooter"],
    "motorcycle": ["bicycle", "scooter"],
    "bird": ["bat", "butterfly", "airplane"],
    "airplane": ["bird", "helicopter", "drone"],
    "horse": ["cow", "donkey", "zebra"],
    "cow": ["horse", "bull", "buffalo"],
    "sheep": ["goat", "lamb", "llama"],
    "chair": ["stool", "bench", "throne"],
    "couch": ["bed", "sofa", "loveseat"],
    "apple": ["orange", "tomato", "ball"],
    "orange": ["apple", "tangerine", "ball"],
    "bottle": ["can", "jar", "vase"],
    "cup": ["mug", "glass", "bowl"],
    "person": ["mannequin", "statue", "doll"],
    "tv": ["monitor", "screen", "painting"],
    "laptop": ["tablet", "book", "notebook"],
    "pizza": ["pie", "cake", "flatbread"],
    "sandwich": ["burger", "wrap", "toast"],
    "bear": ["dog", "gorilla", "person in costume"],
    "elephant": ["hippo", "rhino"],
    "zebra": ["horse", "donkey"],
    "giraffe": ["horse", "deer"],
}

# Color attributes commonly seen
COMMON_COLORS = [
    "red", "blue", "green", "yellow", "orange", "purple", "pink",
    "black", "white", "gray", "brown", "silver", "gold"
]


class COCOAnnotationParser:
    """
    Parser for COCO annotations to support gaslighting prompt generation.
    
    This class provides utilities to extract object information from images
    and generate the data needed for creating gaslighting prompts.
    
    Example:
        >>> parser = COCOAnnotationParser("./data")
        >>> annotations = parser.get_annotations_for_images(image_paths[:100])
        >>> print(annotations[0].objects)  # ['person', 'dog', 'frisbee']
    """
    
    def __init__(
        self, 
        data_dir: str,
        coco_annotations_path: Optional[str] = None
    ):
        """
        Initialize the COCO annotation parser.
        
        Args:
            data_dir: Path to the algonauts data directory
            coco_annotations_path: Optional path to COCO annotations JSON
        """
        self.data_dir = Path(data_dir)
        self.coco_annotations_path = coco_annotations_path
        
        # Cache for loaded annotations
        self._annotations_cache: Dict[int, Dict] = {}
        self._captions_cache: Dict[int, str] = {}
        
        # Try to load COCO annotations if provided
        if coco_annotations_path and Path(coco_annotations_path).exists():
            self._load_coco_annotations(coco_annotations_path)
    
    def _load_coco_annotations(self, path: str):
        """Load COCO annotations from JSON file."""
        logger.info(f"Loading COCO annotations from {path}")
        with open(path, "r") as f:
            data = json.load(f)
        
        # Index by image ID
        if "annotations" in data:
            for ann in data["annotations"]:
                img_id = ann["image_id"]
                if img_id not in self._annotations_cache:
                    self._annotations_cache[img_id] = []
                self._annotations_cache[img_id].append(ann)
        
        if "images" in data:
            self._images_info = {img["id"]: img for img in data["images"]}
        
        logger.info(f"Loaded annotations for {len(self._annotations_cache)} images")
    
    def parse_image_filename(self, filename: str) -> Tuple[int, int]:
        """
        Parse Algonauts image filename to extract indices.
        
        Args:
            filename: e.g., "train-0001_nsd-00013.png"
            
        Returns:
            Tuple of (train_index, nsd_id)
        """
        basename = Path(filename).stem
        parts = basename.split("_")
        
        # Extract train index (1-indexed)
        train_idx = int(parts[0].replace("train-", "").replace("test-", ""))
        
        # Extract NSD ID (0-indexed)
        nsd_id = int(parts[1].replace("nsd-", ""))
        
        return train_idx, nsd_id
    
    def get_confusable_object(self, obj: str) -> str:
        """
        Get a visually similar but different object for gaslighting.
        
        Args:
            obj: The actual object name
            
        Returns:
            A confusable object name
        """
        import random
        
        obj_lower = obj.lower()
        
        if obj_lower in CONFUSABLE_OBJECTS:
            confusables = CONFUSABLE_OBJECTS[obj_lower]
            return random.choice(confusables)
        
        # Fallback: return a random different object from same supercategory
        all_objects = list(COCO_CATEGORIES.values())
        other_objects = [o for o in all_objects if o.lower() != obj_lower]
        return random.choice(other_objects) if other_objects else "unknown object"
    
    def get_wrong_color(self, true_color: Optional[str] = None) -> str:
        """Get a wrong color for attribute gaslighting."""
        import random
        
        available = [c for c in COMMON_COLORS if c != true_color]
        return random.choice(available)
    
    def get_wrong_count(self, true_count: int) -> int:
        """Get an incorrect count for quantity gaslighting."""
        import random
        
        if true_count <= 1:
            return random.choice([2, 3, 4])
        elif true_count <= 3:
            options = [c for c in range(1, 6) if c != true_count]
            return random.choice(options)
        else:
            # For larger counts, offset by 1-3
            offset = random.choice([-3, -2, -1, 1, 2, 3])
            new_count = max(1, true_count + offset)
            return new_count if new_count != true_count else true_count + 1
    
    def extract_objects_from_image_heuristic(
        self, 
        image_path: Path,
        use_filename: bool = True
    ) -> ImageAnnotation:
        """
        Extract object information using heuristics when COCO annotations unavailable.
        
        This is a fallback method that uses the NSD ID and any available metadata.
        For best results, use with actual COCO annotations.
        
        Args:
            image_path: Path to the image
            use_filename: Whether to use filename for NSD ID
            
        Returns:
            ImageAnnotation object with available information
        """
        filename = image_path.name
        train_idx, nsd_id = self.parse_image_filename(filename)
        
        # Create a unique image ID
        image_id = f"{train_idx:05d}"
        
        # Default annotation with placeholder data
        # In production, this should be replaced with actual COCO lookup
        annotation = ImageAnnotation(
            image_id=image_id,
            image_path=image_path,
            nsd_id=nsd_id,
            objects=[],
            object_counts={},
            caption=f"A natural scene image (NSD ID: {nsd_id})"
        )
        
        return annotation
    
    def create_annotation_from_coco(
        self,
        image_path: Path,
        coco_image_id: int,
        annotations: List[Dict],
        caption: str = ""
    ) -> ImageAnnotation:
        """
        Create ImageAnnotation from COCO annotation data.
        
        Args:
            image_path: Path to the image
            coco_image_id: COCO image ID
            annotations: List of COCO annotation dicts for this image
            caption: Optional caption for the image
            
        Returns:
            ImageAnnotation object
        """
        filename = image_path.name
        train_idx, nsd_id = self.parse_image_filename(filename)
        
        # Count objects by category
        object_counts = defaultdict(int)
        objects_list = []
        
        for ann in annotations:
            cat_id = ann.get("category_id")
            if cat_id in COCO_CATEGORIES:
                cat_name = COCO_CATEGORIES[cat_id]
                object_counts[cat_name] += 1
                if cat_name not in objects_list:
                    objects_list.append(cat_name)
        
        return ImageAnnotation(
            image_id=f"{train_idx:05d}",
            image_path=image_path,
            nsd_id=nsd_id,
            objects=objects_list,
            object_counts=dict(object_counts),
            caption=caption if caption else f"An image containing {', '.join(objects_list)}"
        )
    
    def get_annotations_for_images(
        self,
        image_paths: List[Path],
        use_heuristic: bool = True
    ) -> List[ImageAnnotation]:
        """
        Get annotations for a list of images.
        
        Args:
            image_paths: List of image paths
            use_heuristic: Whether to use heuristic fallback
            
        Returns:
            List of ImageAnnotation objects
        """
        annotations = []
        
        for path in image_paths:
            _, nsd_id = self.parse_image_filename(path.name)
            
            # Try to get from cache first
            if nsd_id in self._annotations_cache:
                ann = self.create_annotation_from_coco(
                    path,
                    nsd_id,
                    self._annotations_cache[nsd_id],
                    self._captions_cache.get(nsd_id, "")
                )
            elif use_heuristic:
                ann = self.extract_objects_from_image_heuristic(path)
            else:
                continue
            
            annotations.append(ann)
        
        return annotations
    
    def generate_gaslighting_context(
        self,
        annotation: ImageAnnotation
    ) -> Dict[str, Any]:
        """
        Generate context for gaslighting prompt generation.
        
        Args:
            annotation: ImageAnnotation object
            
        Returns:
            Dictionary with context for prompt generation
        """
        context = {
            "image_id": annotation.image_id,
            "image_path": str(annotation.image_path),
            "nsd_id": annotation.nsd_id,
            "objects": annotation.objects,
            "object_counts": annotation.object_counts,
            "caption": annotation.caption,
            "confusable_objects": {},
            "wrong_counts": {}
        }
        
        # Generate confusable objects
        for obj in annotation.objects:
            context["confusable_objects"][obj] = self.get_confusable_object(obj)
        
        # Generate wrong counts
        for obj, count in annotation.object_counts.items():
            context["wrong_counts"][obj] = self.get_wrong_count(count)
        
        return context
    
    def select_diverse_images(
        self,
        image_paths: List[Path],
        n: int = 200,
        min_objects: int = 1,
        seed: int = 42
    ) -> List[Path]:
        """
        Select a diverse subset of images for gaslighting prompts.
        
        Ensures diversity by:
        1. Filtering images with too few objects
        2. Stratifying by object categories when possible
        
        Args:
            image_paths: All available image paths
            n: Number of images to select
            min_objects: Minimum objects required per image
            seed: Random seed for reproducibility
            
        Returns:
            List of selected image paths
        """
        import random
        random.seed(seed)
        
        # Get annotations for all images
        annotations = self.get_annotations_for_images(image_paths)
        
        # Filter by minimum objects
        valid_annotations = [
            ann for ann in annotations 
            if len(ann.objects) >= min_objects
        ]
        
        if len(valid_annotations) < n:
            logger.warning(
                f"Only {len(valid_annotations)} images have >= {min_objects} objects. "
                f"Using all available."
            )
            return [ann.image_path for ann in valid_annotations]
        
        # Random selection (can be enhanced with stratification)
        selected = random.sample(valid_annotations, n)
        
        return [ann.image_path for ann in selected]
    
    def save_annotations_to_json(
        self,
        annotations: List[ImageAnnotation],
        output_path: str
    ):
        """Save annotations to JSON file."""
        data = []
        for ann in annotations:
            data.append({
                "image_id": ann.image_id,
                "image_path": str(ann.image_path),
                "nsd_id": ann.nsd_id,
                "objects": ann.objects,
                "object_counts": ann.object_counts,
                "caption": ann.caption,
                "attributes": ann.attributes,
                "relationships": ann.relationships
            })
        
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved {len(data)} annotations to {output_path}")
    
    def load_annotations_from_json(
        self,
        json_path: str
    ) -> List[ImageAnnotation]:
        """Load annotations from JSON file."""
        with open(json_path, "r") as f:
            data = json.load(f)
        
        annotations = []
        for item in data:
            ann = ImageAnnotation(
                image_id=item["image_id"],
                image_path=Path(item["image_path"]),
                nsd_id=item["nsd_id"],
                objects=item["objects"],
                object_counts=item["object_counts"],
                caption=item["caption"],
                attributes=item.get("attributes", {}),
                relationships=item.get("relationships", [])
            )
            annotations.append(ann)
        
        return annotations
