import os
from pathlib import Path
from typing import Optional


class Config:
    """Configuration class for the medical image captioning pipeline."""
    
    def __init__(self, 
                 cache_drive: str = "D",
                 project_root: Optional[str] = None,
                 custom_cache_dir: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            cache_drive (str): Drive letter for caching (e.g., "D", "C")
            project_root (str): Root directory of the project
            custom_cache_dir (str): Custom cache directory path
        """
        self.cache_drive = cache_drive.upper()
        self.project_root = Path(project_root) if project_root else Path(__file__).parent.parent
        
        # Set up cache directory
        if custom_cache_dir:
            self.cache_dir = Path(custom_cache_dir)
        else:
            self.cache_dir = Path(f"{self.cache_drive}:/huggingface_cache")
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up all paths
        self._setup_paths()
        self._setup_environment()
    
    def _setup_paths(self):
        """Set up all directory paths."""
        # Main directories
        self.data_dir = self.project_root / "data"
        self.responses_dir = self.project_root / "responses"
        self.vectordb_dir = self.project_root / "vectordb"
        self.cost_analysis_dir = self.project_root / "cost_analysis"
        self.evaluation_dir = self.project_root / "evaluation_results"
        self.pipeline_state_dir = self.project_root / "pipeline_state"
        
        # Create directories
        for dir_path in [self.data_dir, self.responses_dir, self.vectordb_dir, 
                        self.cost_analysis_dir, self.evaluation_dir, self.pipeline_state_dir]:
            dir_path.mkdir(exist_ok=True)
    
    def _setup_environment(self):
        """Set up environment variables for caching."""
        # Set HuggingFace environment variables
        os.environ["HF_HOME"] = str(self.cache_dir)
        os.environ["HF_DATASETS_CACHE"] = str(self.cache_dir)
        os.environ["TRANSFORMERS_CACHE"] = str(self.cache_dir)
        os.environ["HF_HUB_CACHE"] = str(self.cache_dir)
        
        print(f"Configured caching to use {self.cache_drive}: drive: {self.cache_dir}")
    
    def get_vectordb_path(self, name: str = "image_vectordb") -> Path:
        """Get the path for vector database storage."""
        return self.vectordb_dir / name
    
    def get_responses_path(self, timestamp: str) -> Path:
        """Get the path for responses file."""
        return self.responses_dir / f"responses_rag_hf_{timestamp}.jsonl"
    
    def get_cost_analysis_path(self, timestamp: str) -> Path:
        """Get the path for cost analysis report."""
        return self.cost_analysis_dir / f"cost_analysis_{timestamp}.md"
    
    def get_evaluation_path(self, timestamp: str) -> Path:
        """Get the path for evaluation results."""
        return self.evaluation_dir / f"evaluation_results_{timestamp}"
    
    def get_pipeline_state_path(self) -> Path:
        """Get the path for pipeline state."""
        return self.pipeline_state_dir
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            "cache_drive": self.cache_drive,
            "project_root": str(self.project_root),
            "cache_dir": str(self.cache_dir),
            "data_dir": str(self.data_dir),
            "responses_dir": str(self.responses_dir),
            "vectordb_dir": str(self.vectordb_dir),
            "cost_analysis_dir": str(self.cost_analysis_dir),
            "evaluation_dir": str(self.evaluation_dir),
            "pipeline_state_dir": str(self.pipeline_state_dir),
        }


# Global configuration instance
config = Config()

# Convenience function to get config
def get_config() -> Config:
    """Get the global configuration instance."""
    return config

# Function to update configuration
def update_config(cache_drive: str = None, 
                 project_root: str = None, 
                 custom_cache_dir: str = None):
    """Update the global configuration."""
    global config
    config = Config(
        cache_drive=cache_drive or config.cache_drive,
        project_root=project_root or str(config.project_root),
        custom_cache_dir=custom_cache_dir or str(config.cache_dir)
    )
