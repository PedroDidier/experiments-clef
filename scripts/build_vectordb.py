import argparse
from pathlib import Path
from image_vectordb import ImageVectorDB

def main():
    parser = argparse.ArgumentParser(description="Build a vector database from images and captions")
    parser.add_argument("--image_dir", type=str, default="train", 
                        help="Directory containing the training images")
    parser.add_argument("--captions_file", type=str, default="train/train_captions.csv",
                        help="Path to the CSV file containing image captions")
    parser.add_argument("--output_dir", type=str, default="vectordb",
                        help="Directory to save the vector database")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for processing images")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    Path(args.output_dir).mkdir(exist_ok=True, parents=True)
    
    # Initialize vector database
    print(f"Initializing vector database...")
    db = ImageVectorDB()
    
    # Ingest dataset
    print(f"Ingesting images from {args.image_dir} with captions from {args.captions_file}")
    db.ingest_dataset(
        image_dir=args.image_dir,
        captions_file=args.captions_file,
        batch_size=args.batch_size
    )
    
    # Save the database
    print(f"Saving vector database to {args.output_dir}")
    db.save(args.output_dir)
    
    print("Vector database built successfully!")

if __name__ == "__main__":
    main() 