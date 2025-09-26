import os
import random
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
    
    def get_validation_samples(self, num_samples: int = 300, random_seed: int = 42) -> List[Dict[str, Any]]:
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
        sampled_indices = random.sample(range(total_validation), num_samples)
        
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
        Get all training samples for building the vector database with memory-efficient processing.
        
        Returns:
            List[Dict[str, Any]]: List of training samples with image, caption, and image_id
        """
        if self.train_data is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        print(f"Preparing {len(self.train_data)} training samples for vector database")
        print("Using memory-efficient processing...")
        
        samples = []
        for i, sample in enumerate(self.train_data):
            if i % 1000 == 0:
                print(f"Processing training sample {i}/{len(self.train_data)}")
            
            # Extract only the required fields: image, caption, image_id
            sample_data = {
                'image': sample['image'],
                'caption': sample['caption'],
                'image_id': sample['image_id']
            }
            samples.append(sample_data)
            
            # Force garbage collection every 1000 samples
            if i % 1000 == 0:
                import gc
                gc.collect()
        
        print(f"Prepared {len(samples)} training samples for vector database")
        return samples
    
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


if __name__ == "__main__":
    main()
