import os
import random
import csv
from typing import List, Dict, Any, Tuple
from pathlib import Path

import datasets
from datasets import Dataset, DatasetDict
from PIL import Image
import io

from ..config import get_config


class ROCOv2DataHandler:
    """Handler for ROCOv2 dataset from HuggingFace."""
    
    def __init__(self, dataset_name: str = "eltorio/ROCOv2-radiology"):
        """
        Initialize the ROCOv2 data handler.
        
        Args:
            dataset_name (str): Name of the HuggingFace dataset
        """
        self.dataset_name = dataset_name
        self.dataset = None
        self.train_data = None
        self.validation_data = None
        self.test_data = None
        
    def load_dataset(self) -> DatasetDict:
        """
        Load the ROCOv2 dataset from HuggingFace.
        
        Returns:
            DatasetDict: The loaded dataset with train, validation, and test splits
        """
        print(f"Loading ROCOv2 dataset from HuggingFace: {self.dataset_name}")
        
        try:
            # Get configuration
            config = get_config()
            cache_dir = str(config.cache_dir)
            
            print(f"Using cache directory: {cache_dir}")
            
            self.dataset = datasets.load_dataset(
                self.dataset_name,
                cache_dir=cache_dir,
                download_mode="reuse_dataset_if_exists"  # Reuse if already downloaded
            )
            print(f"Successfully loaded dataset with splits: {list(self.dataset.keys())}")
            
            # Store individual splits for easy access
            self.train_data = self.dataset['train']
            self.validation_data = self.dataset['validation'] 
            self.test_data = self.dataset['test']
            
            print(f"Train samples: {len(self.train_data)}")
            print(f"Validation samples: {len(self.validation_data)}")
            print(f"Test samples: {len(self.test_data)}")
            
            return self.dataset
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise
    def get_test_samples(self):
        """
        Get all test samples for evaluation.
        Returns:
            List[Dict[str, Any]]: List of test samples with image, caption, and image_id
        """
        if self.test_data is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        return self.test_data
        
    def get_validation_samples(self, previous_num_samples: int = 0, num_samples: int = 300, random_seed: int = 42) -> List[Dict[str, Any]]:
        """
        Get a random sample of validation images for evaluation.
        
        Args:
            num_samples (int): Number of samples to return
            random_seed (int): Random seed for reproducibility
            
        Returns:
            List[Dict[str, Any]]: List of validation samples with image, caption, and image_id
        """
        if self.validation_data is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        # Set random seed for reproducibility
        random.seed(random_seed)
        
        # Get all validation indices
        total_validation = len(self.validation_data)
        if num_samples > total_validation:
            print(f"Warning: Requested {num_samples} samples but only {total_validation} available. Using all validation samples.")
            num_samples = total_validation
        
        # Randomly sample indices
        previous_sampled_indices = random.sample(range(total_validation), previous_num_samples)
        available_indices = list(set(range(total_validation)) - set(previous_sampled_indices))
        sampled_indices = random.sample(available_indices, num_samples)

        # Extract samples
        samples = []
        for idx in sampled_indices[:10]:
            sample = self.validation_data[idx]
            
            # Extract only the required fields: image, caption, image_id
            sample_data = {
                'image': sample['image'],
                'caption': sample['caption'],
                'image_id': sample['image_id']
            }
            samples.append(sample_data)
        
        print(f"Sampled {len(samples)} validation samples for evaluation")
        return samples
    
    def get_train_samples_for_vectordb(self) -> List[Dict[str, Any]]:
        """
        Get all training samples for building the vector database with memory-efficient processing.
        This method now returns a generator to avoid loading all samples into memory at once.
        
        Returns:
            List[Dict[str, Any]]: List of training samples with image, caption, and image_id
        """
        if self.train_data is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        print(f"Preparing {len(self.train_data)} training samples for vector database")
        print("Using memory-efficient streaming processing...")
        
        # Return the dataset directly instead of converting to list
        # This allows for streaming access without loading everything into memory
        return self.train_data
    
    def get_train_samples_iterator(self, batch_size: int = 1000):
        """
        Get an iterator over training samples for memory-efficient processing.
        
        Args:
            batch_size (int): Number of samples to process in each batch
            
        Yields:
            List[Dict[str, Any]]: Batches of training samples
        """
        if self.train_data is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        total_samples = len(self.train_data)
        print(f"Creating iterator for {total_samples} training samples with batch size {batch_size}")
        
        for i in range(0, total_samples, batch_size):
            batch_end = min(i + batch_size, total_samples)
            print(f"Yielding batch {i//batch_size + 1}/{(total_samples-1)//batch_size + 1} (samples {i}-{batch_end-1})")
            
            # Extract batch samples
            batch_samples = []
            for j in range(i, batch_end):
                sample = self.train_data[j]
                sample_data = {
                    'image': sample['image'],
                    'caption': sample['caption'],
                    'image_id': sample['image_id']
                }
                batch_samples.append(sample_data)
            
            yield batch_samples
    
    def save_sample_images(self, samples: List[Dict[str, Any]], output_dir: str) -> List[str]:
        """
        Save sample images to disk for processing.
        
        Args:
            samples (List[Dict[str, Any]]): List of samples with images
            output_dir (str): Directory to save images
            
        Returns:
            List[str]: List of saved image file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_paths = []
        
        for i, sample in enumerate(samples):
            image = sample['image']
            image_id = sample['image_id']
            
            # Create filename from image_id
            filename = f"{image_id}.jpg"
            filepath = output_path / filename
            
            # Save image
            image.save(filepath, format='JPEG')
            saved_paths.append(str(filepath))
            
            if (i + 1) % 50 == 0:
                print(f"Saved {i + 1}/{len(samples)} images")
        
        print(f"Saved {len(saved_paths)} images to {output_dir}")
        return saved_paths
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded dataset.
        
        Returns:
            Dict[str, Any]: Dataset information
        """
        if self.dataset is None:
            return {"error": "Dataset not loaded"}
        
        info = {
            "dataset_name": self.dataset_name,
            "splits": list(self.dataset.keys()),
            "train_samples": len(self.train_data) if self.train_data else 0,
            "validation_samples": len(self.validation_data) if self.validation_data else 0,
            "test_samples": len(self.test_data) if self.test_data else 0,
            "features": list(self.dataset['train'].features.keys()) if self.train_data else []
        }
        
        return info

class ImageClefDataHandler:
    """Handler for ImageCLEF medical dataset from local files.
    
    Dataset structure:
        data_img_clef/
        ├── natural/
        │   ├── dev_caption/
        │   │   ├── images/ (train + valid images)
        │   │   └── captions.csv
        │   └── test/
        │       └── images/
        └── synth/
            ├── dev_caption_synth/
            │   ├── images/ (train + valid images)
            │   └── captions.csv
            └── test_synth/
                └── images/
    
    Train/valid separation is by filename:
        - train: ImageCLEFmedical_Caption_2026_train_N or ImageCLEFmedical_Caption_2026_synth_train_NNNNNN
        - valid: ImageCLEFmedical_Caption_2026_valid_N or ImageCLEFmedical_Caption_2026_synth_valid_NNNNNN
        - test: ImageCLEFmedical_Caption_2026_test_N or ImageCLEFmedical_Caption_2026_synth_test_NNNNNN
    """
    
    def __init__(self, dataset_type: str = "natural"):
        """
        Initialize the ImageClef data handler.
        
        Args:
            dataset_type (str): Type of dataset - "natural" or "synth"
        """
        self.dataset_type = dataset_type
        self.base_path = self._get_base_path()
        self.captions_df = None
        self.train_data = None
        self.validation_data = None
        self.test_data = None
        
    def _get_base_path(self) -> Path:
        """Get the base path for the dataset."""
        # Get project root (parent of src)
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent
        return project_root / "data_img_clef" / self.dataset_type
    
    def _load_captions(self) -> Dict[str, str]:
        """
        Load captions from CSV file.
        
        Returns:
            Dict[str, str]: Dictionary mapping image_id to caption
        """
        captions_path = self.base_path / "dev_caption" / "captions.csv"
        
        if not captions_path.exists():
            # Try synth path
            if self.dataset_type == "synth":
                captions_path = self.base_path / "dev_caption_synth" / "captions.csv"
            else:
                raise FileNotFoundError(f"Captions file not found: {captions_path}")
        
        captions = {}
        with open(captions_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_id = row['ID'].strip('"')
                caption = row['Caption'].strip('"')
                captions[image_id] = caption
        
        print(f"Loaded {len(captions)} captions from {captions_path.name}")
        return captions
    
    def _classify_split(self, image_id: str) -> str:
        """
        Classify an image into train/valid/test based on filename.
        
        Args:
            image_id (str): Image ID (filename without extension)
            
        Returns:
            str: Split name - "train", "valid", or "test"
        """
        if "synth" in image_id:
            if "_train_" in image_id:
                return "train"
            elif "_valid_" in image_id:
                return "valid"
            elif "_test_" in image_id:
                return "test"
        else:
            if "_train_" in image_id:
                return "train"
            elif "_valid_" in image_id:
                return "valid"
            elif "_test_" in image_id:
                return "test"
        
        return "unknown"
    
    def _extract_sort_key(self, image_id: str) -> Tuple[str, int]:
        """
        Extract a sort key from image_id for natural sorting.
        
        Args:
            image_id (str): Image ID like "ImageCLEFmedical_Caption_2026_train_0" or "ImageCLEFmedical_Caption_2026_synth_train_000000"
            
        Returns:
            Tuple[str, int]: (prefix, numeric_part) for sorting
        """
        # Extract the numeric part at the end
        parts = image_id.rsplit('_', 1)
        if len(parts) == 2:
            prefix = parts[0]
            try:
                num = int(parts[1])
            except ValueError:
                # If not a number, use 0
                num = 0
        else:
            prefix = image_id
            num = 0
        
        return (prefix, num)
    
    def _sort_samples(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Sort samples by image_id with natural sorting (0, 1, 2, 10 instead of 0, 1, 10, 2).
        
        Args:
            samples (List[Dict[str, Any]]): List of samples to sort
            
        Returns:
            List[Dict[str, Any]]: Sorted list
        """
        return sorted(samples, key=lambda x: self._extract_sort_key(x['image_id']))
    
    def load_dataset(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load the ImageCLEF dataset from local files.
        
        Returns:
            Dict[str, List[Dict[str, Any]]]: Dictionary with train, validation, and test splits
        """
        print(f"Loading ImageCLEF dataset (type: {self.dataset_type})")
        
        # Load captions
        self.captions_df = self._load_captions()
        
        # Get images directory paths
        dev_caption_path = self.base_path / "dev_caption"
        if not dev_caption_path.exists():
            dev_caption_path = self.base_path / "dev_caption_synth"
        
        test_path = self.base_path / "test"
        if not test_path.exists():
            test_path = self.base_path / "test_synth"
        
        images_dir = dev_caption_path / "images"
        test_images_dir = test_path / "images"
        
        # Load train and validation data from dev_caption
        self.train_data = []
        self.validation_data = []
        
        if images_dir.exists():
            for img_file in images_dir.iterdir():
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    image_id = img_file.stem
                    split = self._classify_split(image_id)
                    
                    sample = {
                        'image': str(img_file),  # Return as string path
                        'caption': self.captions_df.get(image_id, ""),
                        'image_id': image_id
                    }
                    
                    if split == "train":
                        self.train_data.append(sample)
                    elif split == "valid":
                        self.validation_data.append(sample)
        
        # Load test data
        self.test_data = []
        if test_images_dir.exists():
            for img_file in test_images_dir.iterdir():
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    image_id = img_file.stem
                    
                    sample = {
                        'image': str(img_file),  # Return as string path
                        'caption': self.captions_df.get(image_id, ""),
                        'image_id': image_id
                    }
                    self.test_data.append(sample)
        
        # Sort all partitions naturally
        self.train_data = self._sort_samples(self.train_data)
        self.validation_data = self._sort_samples(self.validation_data)
        self.test_data = self._sort_samples(self.test_data)
        
        print(f"Train samples: {len(self.train_data)}")
        print(f"Validation samples: {len(self.validation_data)}")
        print(f"Test samples: {len(self.test_data)}")
        
        return {
            'train': self.train_data,
            'validation': self.validation_data,
            'test': self.test_data
        }
    
    def get_test_samples(self) -> List[Dict[str, Any]]:
        """
        Get all test samples for evaluation.
        
        Returns:
            List[Dict[str, Any]]: List of test samples with image, caption, and image_id
        """
        if self.test_data is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")

        return self.test_data
    
    def get_validation_samples(self, previous_num_samples: int = 0, num_samples: int = 300, random_seed: int = 42) -> List[Dict[str, Any]]:
        """
        Get a random sample of validation images for evaluation.
        
        Args:
            previous_num_samples (int): Number of previously sampled samples to skip
            num_samples (int): Number of samples to return
            random_seed (int): Random seed for reproducibility
            
        Returns:
            List[Dict[str, Any]]: List of validation samples with image, caption, and image_id
        """
        if self.validation_data is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        random.seed(random_seed)
        
        total_validation = len(self.validation_data)
        if num_samples > total_validation:
            print(f"Warning: Requested {num_samples} samples but only {total_validation} available. Using all validation samples.")
            num_samples = total_validation
        
        # Randomly sample indices
        previous_sampled_indices = random.sample(range(total_validation), min(previous_num_samples, total_validation))
        available_indices = list(set(range(total_validation)) - set(previous_sampled_indices))
        sampled_indices = random.sample(available_indices, min(num_samples, len(available_indices)))

        # Extract samples
        samples = []
        for idx in sampled_indices:
            sample = self.validation_data[idx]
            
            # Extract only the required fields: image, caption, image_id
            sample_data = {
                'image': sample['image'],
                'caption': sample['caption'],
                'image_id': sample['image_id']
            }
            samples.append(sample_data)
        
        print(f"Sampled {len(samples)} validation samples for evaluation")
        return samples
    
    def get_train_samples_for_vectordb(self) -> List[Dict[str, Any]]:
        """
        Get all training samples for building the vector database.
        
        Returns:
            List[Dict[str, Any]]: List of training samples with image, caption, and image_id
        """
        if self.train_data is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        print(f"Preparing {len(self.train_data)} training samples for vector database")
        
        return self.train_data
    
    def get_train_samples_iterator(self, batch_size: int = 1000):
        """
        Get an iterator over training samples for memory-efficient processing.
        
        Args:
            batch_size (int): Number of samples to process in each batch
            
        Yields:
            List[Dict[str, Any]]: Batches of training samples
        """
        if self.train_data is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        total_samples = len(self.train_data)
        print(f"Creating iterator for {total_samples} training samples with batch size {batch_size}")
        
        for i in range(0, total_samples, batch_size):
            batch_end = min(i + batch_size, total_samples)
            print(f"Yielding batch {i//batch_size + 1}/{(total_samples-1)//batch_size + 1} (samples {i}-{batch_end-1})")
            
            # Extract batch samples
            batch_samples = []
            for j in range(i, batch_end):
                sample = self.train_data[j]
                sample_data = {
                    'image': sample['image'],
                    'caption': sample['caption'],
                    'image_id': sample['image_id']
                }
                batch_samples.append(sample_data)
            
            yield batch_samples
    
    def save_sample_images(self, samples: List[Dict[str, Any]], output_dir: str) -> List[str]:
        """
        Save sample images to disk for processing.
        
        Args:
            samples (List[Dict[str, Any]]): List of samples with images
            output_dir (str): Directory to save images
            
        Returns:
            List[str]: List of saved image file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_paths = []
        
        for i, sample in enumerate(samples):
            image_path = sample['image']
            image_id = sample['image_id']
            
            # Create filename from image_id
            filename = f"{image_id}.jpg"
            filepath = output_path / filename
            
            # Copy image
            import shutil
            shutil.copy2(image_path, filepath)
            saved_paths.append(str(filepath))
            
            if (i + 1) % 50 == 0:
                print(f"Saved {i + 1}/{len(samples)} images")
        
        print(f"Saved {len(saved_paths)} images to {output_dir}")
        return saved_paths
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded dataset.
        
        Returns:
            Dict[str, Any]: Dataset information
        """
        if self.train_data is None:
            return {"error": "Dataset not loaded"}
        
        info = {
            "dataset_name": f"ImageCLEF_{self.dataset_type}",
            "dataset_type": self.dataset_type,
            "train_samples": len(self.train_data) if self.train_data else 0,
            "validation_samples": len(self.validation_data) if self.validation_data else 0,
            "test_samples": len(self.test_data) if self.test_data else 0
        }
        
        return info


class UnifiedDataHandler:
    """Wrapper class that returns either ROCOv2DataHandler or ImageClefDataHandler based on the dataset type.
    
    Usage:
        handler = UnifiedDataHandler.get_handler("rocov2")
        handler = UnifiedDataHandler.get_handler("imageclef_natural")
        handler = UnifiedDataHandler.get_handler("imageclef_synth")
    """
    
    HANDLERS = {
        "rocov2": ROCOv2DataHandler,
        "imageclef_natural": lambda: ImageClefDataHandler(dataset_type="natural"),
        "imageclef_synth": lambda: ImageClefDataHandler(dataset_type="synth"),
    }
    
    @classmethod
    def get_handler(cls, dataset_type: str):
        """
        Get the appropriate data handler based on the dataset type.
        
        Args:
            dataset_type (str): Type of dataset - "rocov2", "imageclef_natural", or "imageclef_synth"
            
        Returns:
            Union[ROCOv2DataHandler, ImageClefDataHandler]: The appropriate data handler
            
        Raises:
            ValueError: If dataset_type is not recognized
        """
        if dataset_type not in cls.HANDLERS:
            raise ValueError(f"Unknown dataset type: {dataset_type}. Available: {list(cls.HANDLERS.keys())}")
        
        return cls.HANDLERS[dataset_type]()
    

def main():
    """Example usage of the ROCOv2DataHandler."""
    handler = ROCOv2DataHandler()
    
    # Load dataset
    dataset = handler.load_dataset()
    
    # Get dataset info
    info = handler.get_dataset_info()
    print("Dataset Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Get validation samples
    validation_samples = handler.get_validation_samples(num_samples=10)
    print(f"\nValidation samples (first 3):")
    for i, sample in enumerate(validation_samples[:3]):
        print(f"  Sample {i+1}: {sample['image_id']} - {sample['caption'][:50]}...")
    
    # Get training samples for vectordb
    train_samples = handler.get_train_samples_for_vectordb()
    print(f"\nTraining samples for vectordb: {len(train_samples)}")


def main_imageclef():
    """Example usage of the ImageClefDataHandler."""
    print("=" * 60)
    print("Testing ImageClefDataHandler (natural)")
    print("=" * 60)
    handler = ImageClefDataHandler(dataset_type="natural")
    
    # Load dataset
    dataset = handler.load_dataset()
    
    # Get dataset info
    info = handler.get_dataset_info()
    print("Dataset Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Get validation samples
    validation_samples = handler.get_validation_samples(num_samples=10)
    print(f"\nValidation samples (first 3):")
    for i, sample in enumerate(validation_samples[:3]):
        print(f"  Sample {i+1}: {sample['image_id']} - {sample['caption'][:50]}...")
    
    # Get training samples for vectordb
    train_samples = handler.get_train_samples_for_vectordb()
    print(f"\nTraining samples for vectordb: {len(train_samples)}")


def main_unified():
    """Example usage of the UnifiedDataHandler."""
    print("=" * 60)
    print("Testing UnifiedDataHandler")
    print("=" * 60)
    
    # Test with natural dataset
    print("\n--- Using imageclef_natural ---")
    handler = UnifiedDataHandler.get_handler("imageclef_natural")
    handler.load_dataset()
    info = handler.get_dataset_info()
    print(f"Loaded: {info['train_samples']} train, {info['validation_samples']} valid, {info['test_samples']} test")
    
    # Test with synth dataset
    print("\n--- Using imageclef_synth ---")
    handler = UnifiedDataHandler.get_handler("imageclef_synth")
    handler.load_dataset()
    info = handler.get_dataset_info()
    print(f"Loaded: {info['train_samples']} train, {info['validation_samples']} valid, {info['test_samples']} test")


if __name__ == "__main__":
    main()
    main_imageclef()
    main_unified()
