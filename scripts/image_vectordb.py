import os
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import faiss
import pickle
import json
from typing import List, Dict, Any, Tuple

class ImageVectorDB:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        """
        Initialize the vector database for images.
        
        Args:
            model_name (str): The CLIP model to use for image embeddings
        """
        # Load CLIP model and processor
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        # Initialize FAISS index (will be created when data is added)
        self.index = None
        self.image_metadata = {}
        self.embedding_dim = 512  # Default for CLIP-ViT-B/32
        
    def _get_image_embedding(self, image_path: str) -> np.ndarray:
        """
        Get the embedding for a single image using CLIP.
        
        Args:
            image_path (str): Path to the image
            
        Returns:
            np.ndarray: The image embedding
        """
        try:
            image = Image.open(image_path).convert('RGB')
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)
                
            # Normalize the embedding
            embedding = outputs.cpu().numpy()
            embedding = embedding / np.linalg.norm(embedding)
            return embedding[0]  # Return the first (and only) embedding
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            # Return zeros as a fallback
            return np.zeros(self.embedding_dim, dtype=np.float32)
    
    def ingest_dataset(self, image_dir: str, captions_file: str, batch_size: int = 32) -> None:
        """
        Ingest a dataset of images and their captions into the vector database.
        
        Args:
            image_dir (str): Directory containing the images
            captions_file (str): Path to the CSV file with captions
            batch_size (int): Batch size for processing
        """
        # Load captions
        df = pd.read_csv(captions_file)
        
        # Get all image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(list(Path(image_dir).glob(ext)))
        
        print(f"Found {len(image_files)} images")
        
        # Create FAISS index if not already created
        if self.index is None:
            self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product similarity
        
        # Process images in batches
        all_embeddings = []
        for i in range(0, len(image_files), batch_size):
            batch_files = image_files[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{len(image_files)//batch_size + 1}")
            
            for img_path in batch_files:
                img_name = img_path.name
                
                # Get caption if available
                caption = None
                if 'image_file' in df.columns and 'caption' in df.columns:
                    caption_row = df[df['image_file'] == img_name]
                    if not caption_row.empty:
                        caption = caption_row['caption'].values[0]
                
                # Get embedding
                embedding = self._get_image_embedding(str(img_path))
                all_embeddings.append(embedding)
                
                # Store metadata
                self.image_metadata[len(self.image_metadata)] = {
                    'image_path': str(img_path),
                    'image_name': img_name,
                    'caption': caption
                }
        
        # Add all embeddings to the index
        all_embeddings = np.vstack(all_embeddings).astype(np.float32)
        self.index.add(all_embeddings)
        
        print(f"Added {len(all_embeddings)} image embeddings to the index")
    
    def save(self, save_dir: str) -> None:
        """
        Save the vector database to disk.
        
        Args:
            save_dir (str): Directory to save the database
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, os.path.join(save_dir, "image_index.faiss"))
        
        # Save metadata
        with open(os.path.join(save_dir, "metadata.json"), "w") as f:
            json.dump(self.image_metadata, f)
        
        print(f"Vector database saved to {save_dir}")
    
    def load(self, save_dir: str) -> None:
        """
        Load the vector database from disk.
        
        Args:
            save_dir (str): Directory containing the saved database
        """
        # Load FAISS index
        self.index = faiss.read_index(os.path.join(save_dir, "image_index.faiss"))
        
        # Load metadata
        with open(os.path.join(save_dir, "metadata.json"), "r") as f:
            self.image_metadata = json.load(f)
        
        print(f"Vector database loaded from {save_dir}")
    
    def search_similar_images(self, query_image_path: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar images to the query image.
        
        Args:
            query_image_path (str): Path to the query image
            k (int): Number of similar images to retrieve
            
        Returns:
            List[Dict[str, Any]]: List of similar images with metadata
        """
        # Get embedding for query image
        query_embedding = self._get_image_embedding(query_image_path)
        
        # Search for similar images
        scores, indices = self.index.search(
            np.array([query_embedding]).astype(np.float32), k
        )
        
        # Get metadata for similar images
        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0:  # Valid index
                metadata = self.image_metadata[str(idx)]
                results.append({
                    'image_path': metadata['image_path'],
                    'image_name': metadata['image_name'],
                    'caption': metadata['caption'],
                    'similarity_score': float(scores[0][i])
                })
        
        return results


if __name__ == "__main__":
    # Example usage
    db = ImageVectorDB()
    
    # Ingest dataset
    db.ingest_dataset(
        image_dir="train",
        captions_file="train/train_captions.csv"
    )
    
    # Save the database
    db.save("vectordb")
    
    # Test search
    results = db.search_similar_images("train/ROCOv2_2023_train_000004.jpg")
    for result in results:
        print(f"Image: {result['image_name']}")
        print(f"Caption: {result['caption']}")
        print(f"Similarity: {result['similarity_score']:.4f}")
        print() 