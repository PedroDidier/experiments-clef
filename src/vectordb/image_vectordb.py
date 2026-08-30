import json
import os
import psutil
import gc
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
        
        # Memory monitoring
        self._initial_memory = self._get_memory_usage()
        
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
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return {
            'rss': memory_info.rss / 1024 / 1024,  # Resident Set Size in MB
            'vms': memory_info.vms / 1024 / 1024,  # Virtual Memory Size in MB
        }
    
    def _print_memory_usage(self, stage: str = ""):
        """Print current memory usage."""
        current_memory = self._get_memory_usage()
        memory_increase = current_memory['rss'] - self._initial_memory['rss']
        
        print(f"Memory usage {stage}: RSS={current_memory['rss']:.1f}MB, "
              f"VMS={current_memory['vms']:.1f}MB, "
              f"Increase={memory_increase:.1f}MB")
        
        # Warning if memory usage is getting high
        if current_memory['rss'] > 8000:  # 8GB threshold
            print("⚠️  WARNING: High memory usage detected! Consider reducing batch size.")
    
    def _force_cleanup(self):
        """Force garbage collection and cleanup."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
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

                # Extrai o tensor de dentro do objeto de saida
                if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                    tensor_output = outputs.pooler_output
                elif hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
                    tensor_output = outputs.image_embeds
                elif isinstance(outputs, torch.Tensor):
                    tensor_output = outputs
                else:
                    tensor_output = outputs[0]

            # Normalize the embedding
            embedding = tensor_output.cpu().detach().numpy()
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            return embedding[0]  # Return the first (and only) embedding

        except Exception as e:
            print(f"Error processing image: {e}")
            # Return zeros as a fallback
            return np.zeros(self.embedding_dim, dtype=np.float32)
    
    def build_from_huggingface_dataset(self, train_dataset, batch_size: int = 16, max_samples: int = None):
        """
        Build the vector database from HuggingFace dataset with memory-efficient streaming processing.
        
        Args:
            train_dataset: HuggingFace dataset or iterator over training samples
            batch_size (int): Batch size for processing (reduced for memory efficiency)
            max_samples (int): Maximum number of samples to process (None for all)
        """
        print("Building vector database with memory-efficient streaming processing...")
        print("This will prevent RAM explosion by processing data in small batches...")
        
        # Print initial memory usage
        self._print_memory_usage("before building")
        
        # Create FAISS index
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product similarity
        
        # Process images in streaming batches to avoid memory issues
        all_embeddings = []
        successful_count = 0
        processed_count = 0
        
        # Determine total samples to process
        if hasattr(train_dataset, '__len__'):
            total_samples = len(train_dataset)
            if max_samples:
                total_samples = min(total_samples, max_samples)
        else:
            total_samples = max_samples or "unknown"
        
        print(f"Processing up to {total_samples} training samples in batches of {batch_size}")
        
        # Process in streaming batches
        for i in range(0, total_samples if isinstance(total_samples, int) else 999999, batch_size):
            batch_end = min(i + batch_size, total_samples) if isinstance(total_samples, int) else i + batch_size
            
            # Get batch of samples
            try:
                if hasattr(train_dataset, '__getitem__'):
                    # Handle dataset indexing
                    batch_samples = []
                    for j in range(i, batch_end):
                        if j >= len(train_dataset):
                            break
                        sample = train_dataset[j]
                        batch_samples.append({
                            'image': sample['image'],
                            'caption': sample['caption'],
                            'image_id': sample['image_id']
                        })
                else:
                    # Handle iterator
                    batch_samples = next(train_dataset, [])
                    if not batch_samples:
                        break
            except (IndexError, StopIteration):
                break
            
            if not batch_samples:
                break
                
            print(f"Processing batch {i//batch_size + 1} ({len(batch_samples)} samples) - Total processed: {processed_count}")
            
            batch_embeddings = []
            for sample in batch_samples:
                try:
                    image = sample['image']
                    caption = sample['caption']
                    image_id = sample['image_id']
                    
                    # Get embedding
                    embedding = self._get_image_embedding(image)
                    
                    # Only add if embedding is valid (not all zeros)
                    if np.any(embedding):
                        batch_embeddings.append(embedding)
                        
                        # Store metadata WITHOUT the image to save memory
                        self.image_metadata[successful_count] = {
                            "image_id": image_id,
                            "caption": caption,
                            # Don't store the image in memory - we'll reload it when needed
                        }
                        successful_count += 1
                    else:
                        print(f"Warning: Failed to generate embedding for {image_id}")
                    
                    processed_count += 1
                    
                    # Clear the image from memory immediately
                    del image
                    
                except Exception as e:
                    print(f"Error processing sample {sample.get('image_id', 'unknown')}: {e}")
                    processed_count += 1
                    continue
            
            # Add batch embeddings to the main list
            if batch_embeddings:
                all_embeddings.extend(batch_embeddings)
            
            # Clear batch embeddings from memory
            del batch_embeddings
            
            # Force garbage collection after each batch
            self._force_cleanup()
            
            # Print memory usage every 10 batches
            if (i // batch_size) % 10 == 0:
                self._print_memory_usage(f"after batch {i//batch_size + 1}")
            
            # Check if we've reached max_samples
            if max_samples and processed_count >= max_samples:
                break
        
        if not all_embeddings:
            raise ValueError(
                "No valid embeddings were generated. Check your images and captions."
            )
        
        print(f"Adding {len(all_embeddings)} embeddings to FAISS index...")
        
        # Add all embeddings to the index in chunks to avoid memory issues
        chunk_size = 10000  # Process embeddings in chunks
        for i in range(0, len(all_embeddings), chunk_size):
            chunk = all_embeddings[i:i + chunk_size]
            chunk_array = np.vstack(chunk).astype(np.float32)
            self.index.add(chunk_array)
            del chunk_array
        
        # Clear embeddings from memory
        del all_embeddings
        self._force_cleanup()
        
        # Print final memory usage
        self._print_memory_usage("after building")
        
        print(f"Successfully added {successful_count} image embeddings to the index")
        print(f"Total samples processed: {processed_count}")
        print("  - Images not stored in memory for efficiency")
        print("  - Vector database ready for similarity search")
    
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
            if idx >= 0 and idx in self.image_metadata:  # Valid index - use integer key
                metadata = self.image_metadata[idx]
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
        
        # Load metadata and convert string keys back to integers
        with open(metadata_path, "r") as f:
            loaded_metadata = json.load(f)
        
        # Convert string keys back to integers
        self.image_metadata = {int(k): v for k, v in loaded_metadata.items()}
        
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
