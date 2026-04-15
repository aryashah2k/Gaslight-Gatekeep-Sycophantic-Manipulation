"""
Script to download NSD-to-COCO mapping and COCO annotations.

This script downloads the necessary files to map NSD IDs (from Algonauts images)
to COCO image IDs and their annotations.

Required files:
1. nsd_stim_info_merged.csv - NSD stimulus info with COCO ID mapping
2. COCO instances annotations - Object annotations (instances_train2017.json, instances_val2017.json)
3. COCO captions annotations - Image captions (captions_train2017.json, captions_val2017.json)

Usage:
    python scripts/download_coco_mapping.py
    python scripts/download_coco_mapping.py --output-dir data/coco_annotations
"""

import os
import sys
import argparse
import urllib.request
import zipfile
from pathlib import Path
import json
import logging

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# URLs for required files
NSD_STIM_INFO_URL = "https://natural-scenes-dataset.s3.amazonaws.com/nsddata/experiments/nsd/nsd_stim_info_merged.csv"

COCO_ANNOTATIONS_URLS = {
    "instances": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    "captions": None  # Included in the same zip
}


def download_file(url: str, output_path: Path, desc: str = "Downloading"):
    """Download file with progress."""
    from tqdm import tqdm
    
    logger.info(f"{desc}: {url}")
    
    try:
        response = urllib.request.urlopen(url)
        total_size = int(response.headers.get('content-length', 0))
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=output_path.name) as pbar:
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        logger.info(f"Saved to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False


def extract_zip(zip_path: Path, extract_to: Path):
    """Extract zip file."""
    logger.info(f"Extracting {zip_path} to {extract_to}")
    
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_to)
    
    logger.info("Extraction complete")


def create_nsd_to_coco_mapping(csv_path: Path, output_path: Path):
    """Create NSD to COCO ID mapping from NSD stim info CSV."""
    import pandas as pd
    
    logger.info(f"Loading NSD stim info from {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Create mapping: nsdId -> cocoId
    mapping = {}
    for _, row in df.iterrows():
        nsd_id = int(row['nsdId'])
        coco_id = int(row['cocoId'])
        coco_split = row.get('cocoSplit', 'train2017')
        
        mapping[nsd_id] = {
            'coco_id': coco_id,
            'coco_split': coco_split,
        }
    
    # Save mapping
    with open(output_path, 'w') as f:
        json.dump(mapping, f)
    
    logger.info(f"Created mapping for {len(mapping)} NSD images -> {output_path}")
    return mapping


def create_coco_annotations_index(annotations_dir: Path, output_path: Path):
    """Create indexed COCO annotations for fast lookup."""
    logger.info("Creating COCO annotations index...")
    
    # Load annotations from both train and val
    all_annotations = {}  # coco_id -> list of annotations
    all_captions = {}     # coco_id -> list of captions
    categories = {}       # category_id -> category info
    
    for split in ['train2017', 'val2017']:
        # Load instances
        instances_path = annotations_dir / 'annotations' / f'instances_{split}.json'
        if instances_path.exists():
            logger.info(f"Loading {instances_path}")
            with open(instances_path, 'r') as f:
                data = json.load(f)
            
            # Index categories
            for cat in data.get('categories', []):
                categories[cat['id']] = cat
            
            # Index annotations by image_id
            for ann in data.get('annotations', []):
                img_id = ann['image_id']
                if img_id not in all_annotations:
                    all_annotations[img_id] = []
                all_annotations[img_id].append({
                    'category_id': ann['category_id'],
                    'bbox': ann.get('bbox'),
                    'area': ann.get('area')
                })
        
        # Load captions
        captions_path = annotations_dir / 'annotations' / f'captions_{split}.json'
        if captions_path.exists():
            logger.info(f"Loading {captions_path}")
            with open(captions_path, 'r') as f:
                data = json.load(f)
            
            for ann in data.get('annotations', []):
                img_id = ann['image_id']
                if img_id not in all_captions:
                    all_captions[img_id] = []
                all_captions[img_id].append(ann.get('caption', ''))
    
    # Create indexed output
    index = {
        'categories': categories,
        'annotations': {str(k): v for k, v in all_annotations.items()},
        'captions': {str(k): v for k, v in all_captions.items()},
        'stats': {
            'n_images_with_annotations': len(all_annotations),
            'n_images_with_captions': len(all_captions),
            'n_categories': len(categories)
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(index, f)
    
    logger.info(f"Created index with {len(all_annotations)} images -> {output_path}")
    return index


def main():
    parser = argparse.ArgumentParser(description="Download COCO mapping files")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/coco_annotations",
        help="Output directory for downloaded files"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading, just create indexes from existing files"
    )
    
    args = parser.parse_args()
    
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("COCO Mapping Setup for Algonauts")
    logger.info("=" * 60)
    
    # Step 1: Download NSD stim info
    nsd_csv_path = output_dir / "nsd_stim_info_merged.csv"
    nsd_mapping_path = output_dir / "nsd_to_coco_mapping.json"
    
    if not args.skip_download:
        if not nsd_csv_path.exists():
            logger.info("\n[1/3] Downloading NSD stimulus info...")
            if not download_file(NSD_STIM_INFO_URL, nsd_csv_path, "NSD stim info"):
                logger.error("Failed to download NSD stim info. Please download manually from:")
                logger.error(f"  {NSD_STIM_INFO_URL}")
                logger.error(f"  Save to: {nsd_csv_path}")
                return 1
        else:
            logger.info(f"[1/3] NSD stim info already exists: {nsd_csv_path}")
    
    # Create NSD to COCO mapping
    if nsd_csv_path.exists() and not nsd_mapping_path.exists():
        try:
            create_nsd_to_coco_mapping(nsd_csv_path, nsd_mapping_path)
        except Exception as e:
            logger.error(f"Failed to create mapping: {e}")
    
    # Step 2: Download COCO annotations
    coco_zip_path = output_dir / "annotations_trainval2017.zip"
    coco_annotations_dir = output_dir
    
    if not args.skip_download:
        annotations_exist = (
            (coco_annotations_dir / "annotations" / "instances_train2017.json").exists() or
            (coco_annotations_dir / "annotations" / "instances_val2017.json").exists()
        )
        
        if not annotations_exist:
            logger.info("\n[2/3] Downloading COCO annotations (~252MB)...")
            logger.info("This may take a while...")
            
            if not download_file(COCO_ANNOTATIONS_URLS["instances"], coco_zip_path, "COCO annotations"):
                logger.error("Failed to download COCO annotations. Please download manually from:")
                logger.error(f"  {COCO_ANNOTATIONS_URLS['instances']}")
                logger.error(f"  Extract to: {coco_annotations_dir}")
                return 1
            
            # Extract
            extract_zip(coco_zip_path, coco_annotations_dir)
            
            # Cleanup zip
            coco_zip_path.unlink()
        else:
            logger.info(f"[2/3] COCO annotations already exist")
    
    # Step 3: Create indexed annotations
    coco_index_path = output_dir / "coco_annotations_index.json"
    
    if (coco_annotations_dir / "annotations").exists():
        logger.info("\n[3/3] Creating COCO annotations index...")
        try:
            create_coco_annotations_index(coco_annotations_dir, coco_index_path)
        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SETUP COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir}")
    
    files_created = []
    if nsd_mapping_path.exists():
        files_created.append(f"  - {nsd_mapping_path.name} (NSD -> COCO ID mapping)")
    if coco_index_path.exists():
        files_created.append(f"  - {coco_index_path.name} (COCO annotations index)")
    
    if files_created:
        logger.info("Files created:")
        for f in files_created:
            logger.info(f)
    
    logger.info("\nYou can now run prompt generation:")
    logger.info("  python scripts/03_generate_prompts.py --n-images 200")
    
    return 0


if __name__ == "__main__":
    from tqdm import tqdm
    sys.exit(main())
