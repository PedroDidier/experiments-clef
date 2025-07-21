import argparse
from pathlib import Path

from .image_vectordb import ImageVectorDB


def main():
    parser = argparse.ArgumentParser(
        description="Build a vector database from images and captions"
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="train",
        help="Directory containing the training images",
    )
    parser.add_argument(
        "--captions_file",
        type=str,
        default="train_captions.csv",
        help="Path to the CSV file containing image captions",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="vectordb",
        help="Directory to save the vector database",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for processing images"
    )
    args = parser.parse_args()

    # Validate inputs
    image_dir_path = Path(args.image_dir)
    captions_file_path = Path(args.captions_file)

    if not image_dir_path.exists():
        raise FileNotFoundError(f"Image directory not found: {args.image_dir}")

    if not captions_file_path.exists():
        raise FileNotFoundError(f"Captions file not found: {args.captions_file}")

    # Create output directory if it doesn't exist
    Path(args.output_dir).mkdir(exist_ok=True, parents=True)

    # Initialize vector database
    print(f"Initializing vector database...")
    try:
        db = ImageVectorDB()
    except Exception as e:
        print(f"Failed to initialize vector database: {e}")
        return

    # Ingest dataset
    print(
        f"Ingesting images from {args.image_dir} with captions from {args.captions_file}"
    )
    try:
        db.ingest_dataset(
            image_dir=args.image_dir,
            captions_file=args.captions_file,
            batch_size=args.batch_size,
        )
    except Exception as e:
        print(f"Failed to ingest dataset: {e}")
        return

    # Save the database
    print(f"Saving vector database to {args.output_dir}")
    try:
        db.save(args.output_dir)
        print("Vector database built successfully!")
    except Exception as e:
        print(f"Failed to save vector database: {e}")
        return


if __name__ == "__main__":
    main()
