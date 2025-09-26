import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from ..config import get_config


class ImageVectorDB:
    """Vector database for medical image similarity search using CLIP embeddings."""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """
        Initialize the vector database for images.
        
        Args:
            model_name (str): The CLIP model to use for image embeddings
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load CLIP model and processor
        self._load_clip_model()
        
        # Initialize FAISS index (will be created when data is added)
        self.index = None
        self.image_metadata = {}
        self.embedding_dim = 512  # Default for CLIP-ViT-B/32
        
    def _load_clip_model(self):
        """Load the CLIP model and processor."""
        try:
            # Get configuration
            config = get_config()
            cache_dir = str(config.cache_dir)
            
            print(f"Loading CLIP model: {self.model_name}")
            print(f"Using cache directory: {cache_dir}")
            
            self.model = CLIPModel.from_pretrained(
                self.model_name, 
                cache_dir=cache_dir
            ).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(
                self.model_name, 
                cache_dir=cache_dir
            )
            print(f"Successfully loaded CLIP model: {self.model_name}")
        except Exception as e:
            print(f"Error loading CLIP model {self.model_name}: {e}")
            print("Trying alternative model: openai/clip-vit-base-patch16")
            try:
                config = get_config()
                cache_dir = str(config.cache_dir)
                
                self.model = CLIPModel.from_pretrained(
                    "openai/clip-vit-base-patch16", 
                    cache_dir=cache_dir
                ).to(self.device)
                self.processor = CLIPProcessor.from_pretrained(
                    "openai/clip-vit-base-patch16", 
                    cache_dir=cache_dir
                )
                print("Successfully loaded alternative CLIP model")
            except Exception as e2:
                print(f"Error loading alternative CLIP model: {e2}")
                raise RuntimeError(
                    f"Failed to load any CLIP model. Original error: {e}, Alternative error: {e2}"
                )
    
    def _get_image_embedding(self, image: Image.Image) -> np.ndarray:
        """
        Get the embedding for a single image using CLIP.
        
        Args:
            image (Image.Image): PIL Image object
            
        Returns:
            np.ndarray: The image embedding
        """
        try:
            # Convert to RGB if needed
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)
            
            # Normalize the embedding
            embedding = outputs.cpu().numpy()
            embedding = embedding / np.linalg.norm(embedding)
            return embedding[0]  # Return the first (and only) embedding
        except Exception as e:
            print(f"Error processing image: {e}")
            # Return zeros as a fallback
            return np.zeros(self.embedding_dim, dtype=np.float32)
    
    def build_from_huggingface_dataset(self, train_samples: List[Dict[str, Any]], batch_size: int = 16):
        """
        Build the vector database from HuggingFace dataset samples with memory-efficient processing.
        
        Args:
            train_samples (List[Dict[str, Any]]): List of training samples with image, caption, image_id
            batch_size (int): Batch size for processing (reduced for memory efficiency)
        """
        print(f"Building vector database from {len(train_samples)} training samples")
        print("Using memory-efficient processing to avoid RAM overload...")
        
        # Create FAISS index
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product similarity
        
        # Process images in smaller batches to avoid memory issues
        all_embeddings = []
        successful_count = 0
        
        for i in range(0, len(train_samples), batch_size):
            batch_samples = train_samples[i : i + batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(train_samples)-1)//batch_size + 1} ({len(batch_samples)} samples)")
            
            for sample in batch_samples:
                image = sample['image']
                caption = sample['caption']
                image_id = sample['image_id']
                
                # Get embedding
                embedding = self._get_image_embedding(image)
                
                # Only add if embedding is valid (not all zeros)
                if np.any(embedding):
                    all_embeddings.append(embedding)
                    
                    # Store metadata WITHOUT the image to save memory
                    self.image_metadata[successful_count] = {
                        "image_id": image_id,
                        "caption": caption,
                        # Don't store the image in memory - we'll reload it when needed
                    }
                    successful_count += 1
                else:
                    print(f"Warning: Failed to generate embedding for {image_id}")
                
                # Clear the image from memory immediately
                del image
            
            # Force garbage collection after each batch
            import gc
            gc.collect()
        
        if not all_embeddings:
            raise ValueError(
                "No valid embeddings were generated. Check your images and captions."
            )
        
        # Add all embeddings to the index
        all_embeddings = np.vstack(all_embeddings).astype(np.float32)
        self.index.add(all_embeddings)
        
        # Clear embeddings from memory
        del all_embeddings
        import gc
        gc.collect()
        
        print(f"Successfully added {successful_count} image embeddings to the index")
        print("  - Images not stored in memory for efficiency")
    
    def build_from_image_files(self, image_dir: str, captions_file: str, batch_size: int = 32):
        """
        Build the vector database from image files and captions CSV (legacy method).
        
        Args:
            image_dir (str): Directory containing the images
            captions_file (str): Path to the CSV file with captions
            batch_size (int): Batch size for processing
        """
        import pandas as pd
        
        # Load captions
        df = pd.read_csv(captions_file)
        print(f"Loaded captions file with {len(df)} entries")
        
        # Get all image files
        image_files = []
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            image_files.extend(list(Path(image_dir).glob(ext)))
        
        print(f"Found {len(image_files)} images in {image_dir}")
        
        # Create FAISS index if not already created
        if self.index is None:
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        
        # Process images in batches
        all_embeddings = []
        successful_count = 0
        
        for i in range(0, len(image_files), batch_size):
            batch_files = image_files[i : i + batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(image_files)-1)//batch_size + 1} ({len(batch_files)} images)")
            
            for img_path in batch_files:
                img_name = img_path.name
                
                # Get caption if available
                caption = None
                if "image_file" in df.columns and "caption" in df.columns:
                    caption_row = df[df["image_file"] == img_name]
                    if not caption_row.empty:
                        caption = caption_row["caption"].values[0]
                    else:
                        print(f"Warning: No caption found for {img_name}")
                        continue  # Skip images without captions for RAG
                
                # Load and process image
                try:
                    image = Image.open(img_path).convert("RGB")
                    embedding = self._get_image_embedding(image)
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
                    continue
                
                # Only add if embedding is valid (not all zeros)
                if np.any(embedding):
                    all_embeddings.append(embedding)
                    
                    # Store metadata
                    self.image_metadata[successful_count] = {
                        "image_path": str(img_path),
                        "image_name": img_name,
                        "caption": caption,
                    }
                    successful_count += 1
                else:
                    print(f"Warning: Failed to generate embedding for {img_name}")
        
        if not all_embeddings:
            raise ValueError(
                "No valid embeddings were generated. Check your images and captions."
            )
        
        # Add all embeddings to the index
        all_embeddings = np.vstack(all_embeddings).astype(np.float32)
        self.index.add(all_embeddings)
        
        print(f"Successfully added {len(all_embeddings)} image embeddings to the index")
    
    def search_similar_images(self, query_image: Image.Image, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar images to the query image.
        
        Args:
            query_image (Image.Image): PIL Image object to search for
            k (int): Number of similar images to retrieve
            
        Returns:
            List[Dict[str, Any]]: List of similar images with metadata
        """
        if self.index is None:
            raise ValueError(
                "Vector database not loaded. Please build the database first."
            )
        
        # Get embedding for query image
        query_embedding = self._get_image_embedding(query_image)
        
        if not np.any(query_embedding):
            print("Warning: Failed to generate embedding for query image")
            return []
        
        # Search for similar images
        scores, indices = self.index.search(
            np.array([query_embedding]).astype(np.float32), k
        )
        
        # Get metadata for similar images (without images to save memory)
        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0 and str(idx) in self.image_metadata:  # Valid index
                metadata = self.image_metadata[str(idx)]
                results.append(
                    {
                        "image_id": metadata.get("image_id", f"unknown_{idx}"),
                        "caption": metadata["caption"],
                        "similarity_score": float(scores[0][i]),
                        # Don't include images to save memory
                        "image_path": metadata.get("image_path"),  # Include path if available
                    }
                )
        
        return results
    
    def save(self, save_dir: str = None) -> None:
        """
        Save the vector database to disk.
        
        Args:
            save_dir (str): Directory to save the database (uses config if None)
        """
        if save_dir is None:
            config = get_config()
            save_dir = str(config.get_vectordb_path())
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Save FAISS index
        index_path = os.path.join(save_dir, "image_index.faiss")
        faiss.write_index(self.index, index_path)
        
        # Save metadata (without PIL images as they can't be serialized)
        metadata_to_save = {}
        for idx, metadata in self.image_metadata.items():
            metadata_copy = metadata.copy()
            if "image" in metadata_copy:
                del metadata_copy["image"]  # Remove PIL image
            metadata_to_save[idx] = metadata_copy
        
        metadata_path = os.path.join(save_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata_to_save, f, indent=2)
        
        print(f"Vector database saved to {save_dir}")
        print(f"  - Index: {index_path}")
        print(f"  - Metadata: {metadata_path}")
    
    def load(self, save_dir: str = None) -> None:
        """
        Load the vector database from disk.
        
        Args:
            save_dir (str): Directory containing the saved database (uses config if None)
        """
        if save_dir is None:
            config = get_config()
            save_dir = str(config.get_vectordb_path())
        
        index_path = os.path.join(save_dir, "image_index.faiss")
        metadata_path = os.path.join(save_dir, "metadata.json")
        
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"Vector database index not found at {index_path}. Please build the database first."
            )
        
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"Vector database metadata not found at {metadata_path}. Please build the database first."
            )
        
        # Load FAISS index
        self.index = faiss.read_index(index_path)
        
        # Load metadata
        with open(metadata_path, "r") as f:
            self.image_metadata = json.load(f)
        
        print(f"Vector database loaded from {save_dir}")
        print(f"  - Index contains {self.index.ntotal} embeddings")
        print(f"  - Metadata contains {len(self.image_metadata)} entries")


def main():
    """Example usage of the ImageVectorDB."""
    db = ImageVectorDB()
    
    # Example with HuggingFace dataset
    from src.data.dataset import ROCOv2DataHandler
    
    handler = ROCOv2DataHandler()
    handler.load_dataset()
    
    # Get training samples
    train_samples = handler.get_train_samples_for_vectordb()
    
    # Build vector database
    db.build_from_huggingface_dataset(train_samples[:100])  # Use first 100 for testing
    
    # Test search
    if train_samples:
        test_image = train_samples[0]['image']
        results = db.search_similar_images(test_image, k=3)
        for result in results:
            print(f"Image ID: {result['image_id']}")
            print(f"Caption: {result['caption']}")
            print(f"Similarity: {result['similarity_score']:.4f}")
            print()


if __name__ == "__main__":
    main()
