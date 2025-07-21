import json
import os
from pathlib import Path
from typing import Any, Dict, List

import faiss
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


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

        try:
            print(f"Loading CLIP model: {model_name}")
            self.model = CLIPModel.from_pretrained(model_name, cache_dir=".cache").to(
                self.device
            )
            self.processor = CLIPProcessor.from_pretrained(
                model_name, cache_dir=".cache"
            )
            print(f"Successfully loaded CLIP model: {model_name}")
        except Exception as e:
            print(f"Error loading CLIP model {model_name}: {e}")
            print("Trying alternative model: openai/clip-vit-base-patch16")
            try:
                self.model = CLIPModel.from_pretrained(
                    "openai/clip-vit-base-patch16", cache_dir=".cache"
                ).to(self.device)
                self.processor = CLIPProcessor.from_pretrained(
                    "openai/clip-vit-base-patch16", cache_dir=".cache"
                )
                print("Successfully loaded alternative CLIP model")
            except Exception as e2:
                print(f"Error loading alternative CLIP model: {e2}")
                raise RuntimeError(
                    f"Failed to load any CLIP model. Original error: {e}, Alternative error: {e2}"
                )

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
            image = Image.open(image_path).convert("RGB")
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

    def ingest_dataset(
        self, image_dir: str, captions_file: str, batch_size: int = 32
    ) -> None:
        """
        Ingest a dataset of images and their captions into the vector database.
        
        Args:
            image_dir (str): Directory containing the images
            captions_file (str): Path to the CSV file with captions
            batch_size (int): Batch size for processing
        """
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
            self.index = faiss.IndexFlatIP(
                self.embedding_dim
            )  # Inner product similarity

        # Process images in batches
        all_embeddings = []
        successful_count = 0

        for i in range(0, len(image_files), batch_size):
            batch_files = image_files[i : i + batch_size]
            print(
                f"Processing batch {i//batch_size + 1}/{(len(image_files)-1)//batch_size + 1} ({len(batch_files)} images)"
            )

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

                # Get embedding
                embedding = self._get_image_embedding(str(img_path))

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

    def save(self, save_dir: str) -> None:
        """
        Save the vector database to disk.
        
        Args:
            save_dir (str): Directory to save the database
        """
        os.makedirs(save_dir, exist_ok=True)

        # Save FAISS index
        index_path = os.path.join(save_dir, "image_index.faiss")
        faiss.write_index(self.index, index_path)

        # Save metadata
        metadata_path = os.path.join(save_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(self.image_metadata, f, indent=2)

        print(f"Vector database saved to {save_dir}")
        print(f"  - Index: {index_path}")
        print(f"  - Metadata: {metadata_path}")

    def load(self, save_dir: str) -> None:
        """
        Load the vector database from disk.
        
        Args:
            save_dir (str): Directory containing the saved database
        """
        index_path = os.path.join(save_dir, "image_index.faiss")
        metadata_path = os.path.join(save_dir, "metadata.json")

        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"Vector database index not found at {index_path}. Please build the database first using build_vectordb.py"
            )

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"Vector database metadata not found at {metadata_path}. Please build the database first using build_vectordb.py"
            )

        # Load FAISS index
        self.index = faiss.read_index(index_path)

        # Load metadata
        with open(metadata_path, "r") as f:
            self.image_metadata = json.load(f)

        print(f"Vector database loaded from {save_dir}")
        print(f"  - Index contains {self.index.ntotal} embeddings")
        print(f"  - Metadata contains {len(self.image_metadata)} entries")

    def search_similar_images(
        self, query_image_path: str, k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar images to the query image.
        
        Args:
            query_image_path (str): Path to the query image
            k (int): Number of similar images to retrieve
            
        Returns:
            List[Dict[str, Any]]: List of similar images with metadata
        """
        if self.index is None:
            raise ValueError(
                "Vector database not loaded. Please load the database first."
            )

        # Get embedding for query image
        query_embedding = self._get_image_embedding(query_image_path)

        if not np.any(query_embedding):
            print(
                f"Warning: Failed to generate embedding for query image {query_image_path}"
            )
            return []

        # Search for similar images
        scores, indices = self.index.search(
            np.array([query_embedding]).astype(np.float32), k
        )

        # Get metadata for similar images
        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0 and str(idx) in self.image_metadata:  # Valid index
                metadata = self.image_metadata[str(idx)]
                results.append(
                    {
                        "image_path": metadata["image_path"],
                        "image_name": metadata["image_name"],
                        "caption": metadata["caption"],
                        "similarity_score": float(scores[0][i]),
                    }
                )

        return results


if __name__ == "__main__":
    # Example usage
    db = ImageVectorDB()

    # Ingest dataset
    db.ingest_dataset(image_dir="train", captions_file="train/train_captions.csv")

    # Save the database
    db.save("vectordb")

    # Test search
    results = db.search_similar_images("train/ROCOv2_2023_train_000004.jpg")
    for result in results:
        print(f"Image: {result['image_name']}")
        print(f"Caption: {result['caption']}")
        print(f"Similarity: {result['similarity_score']:.4f}")
        print()
