import os
import yaml
from pathlib import Path
from typing import Optional, Dict, Any


class Config:
    """Configuration class for the medical image captioning pipeline."""
    
    def __init__(self, 
                 cache_drive: str = "D",
                 project_root: Optional[str] = None,
                 custom_cache_dir: Optional[str] = None,
                 config_file: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            cache_drive (str): Drive letter for caching (e.g., "D", "C")
            project_root (str): Root directory of the project
            custom_cache_dir (str): Custom cache directory path
            config_file (str): Path to YAML configuration file
        """
        self.project_root = Path(project_root) if project_root else Path(__file__).parent.parent

        # Load configuration from file if provided
        self.config_file = config_file
        self.config_data = self._load_config_file(config_file)
        
        # Set up cache configuration
        self.cache_drive = self.config_data.get('cache', {}).get('drive', cache_drive).upper()
        custom_cache = self.config_data.get('cache', {}).get('custom_dir') or custom_cache_dir
        
        if custom_cache:
            self.cache_dir = Path(custom_cache).expanduser()
        elif os.name == "nt":
            # Windows: honour the configured drive letter.
            self.cache_dir = Path(f"{self.cache_drive}:/huggingface_cache")
        else:
            # POSIX has no drive letters; a "D:/..." path would create a
            # literal "D:" directory inside the working tree.
            self.cache_dir = Path(
                os.getenv("HF_HOME") or Path.home() / ".cache" / "huggingface"
            )
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up all paths
        self._setup_paths()
        self._setup_environment()
    
    def _load_config_file(self, config_file: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not config_file:
            # Try to find config files in order of preference
            config_files = [
                self.project_root / "config_local.yaml",
                self.project_root / "config.yaml"
            ]
            
            for config_path in config_files:
                if config_path.exists():
                    config_file = str(config_path)
                    break
        
        if config_file and Path(config_file).exists():
            try:
                with open(config_file, 'r') as f:
                    return yaml.safe_load(f) or {}
            except Exception as e:
                print(f"Warning: Could not load config file {config_file}: {e}")
                return {}
        
        return {}
    
    def _setup_paths(self):
        """Set up all directory paths."""
        # Get output configuration
        output_config = self.config_data.get('output', {})
        
        # Main directories
        self.data_dir = self.project_root / "data"
        self.responses_dir = self.project_root / output_config.get('responses_dir', 'responses')
        self.vectordb_dir = self.project_root / output_config.get('vectordb_dir', 'vectordb')
        self.cost_analysis_dir = self.project_root / output_config.get('cost_analysis_dir', 'cost_analysis')
        self.evaluation_dir = self.project_root / output_config.get('evaluation_dir', 'evaluation_results')
        self.pipeline_state_dir = self.project_root / output_config.get('pipeline_state_dir', 'pipeline_state')
        
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
        
        print(f"Configured HuggingFace cache directory: {self.cache_dir}")
    
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
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation (e.g., 'model.name')."""
        keys = key.split('.')
        value = self.config_data
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self.config_data.get('model', {})
    
    def get_dataset_config(self) -> Dict[str, Any]:
        """Get dataset configuration."""
        return self.config_data.get('dataset', {})
    
    def get_rag_config(self) -> Dict[str, Any]:
        """Get RAG configuration."""
        return self.config_data.get('rag', {})
    
    def get_memory_config(self) -> Dict[str, Any]:
        """Get memory configuration."""
        return self.config_data.get('memory', {})
    
    def get_analysis_config(self) -> Dict[str, Any]:
        """Get analysis configuration."""
        return self.config_data.get('analysis', {})
    
    def get_prompt_config(self) -> Dict[str, Any]:
        """Get prompt configuration."""
        return self.config_data.get('prompt', {})
    
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
            "config_data": self.config_data,
        }
    
    def update_model_config(self, new_config: Dict[str, Any]):
        """Update model configuration."""
        if 'model' not in self.config_data:
            self.config_data['model'] = {}
        
        for key, value in new_config.items():
            self.config_data['model'][key] = value

    def update_rag_config(self, new_config: Dict[str, Any]):
        """Update RAG configuration."""
        if 'rag' not in self.config_data:
            self.config_data['rag'] = {}
        
        for key, value in new_config.items():
            self.config_data['rag'][key] = value
    
    def update_prompt_config(self, new_config: Dict[str, Any]):
        """Update prompt configuration."""
        if 'prompt' not in self.config_data:
            self.config_data['prompt'] = {}
        
        for key, value in new_config.items():
            self.config_data['prompt'][key] = value
        
    def update_dataset_config(self, new_config: Dict[str, Any]):
        """Update dataset configuration."""
        if 'dataset' not in self.config_data:
            self.config_data['dataset'] = {}
        
        for key, value in new_config.items():
            self.config_data['dataset'][key] = value


# Global configuration instance
config = Config(config_file="config_deepinfra.yaml")

# Convenience function to get config
def get_config() -> Config:
    """Get the global configuration instance."""
    return config

def set_config(config_file: str) -> Config:
    """Replace the global configuration with one loaded from ``config_file``.

    Call this before building the pipeline: ``dataset`` and ``vectordb`` read
    the global via ``get_config()`` at call time, so reassigning it here is
    picked up by every component.
    """
    global config
    config = Config(config_file=config_file)
    return config


# Function to update configuration
def update_config(cache_drive: str = None,
                 project_root: str = None,
                 custom_cache_dir: str = None,
                 config_file: str = None):
    """Update the global configuration, preserving the loaded YAML file."""
    global config
    config = Config(
        cache_drive=cache_drive or config.cache_drive,
        project_root=project_root or str(config.project_root),
        custom_cache_dir=custom_cache_dir,
        # Without this the YAML would be silently discarded and every
        # setting would fall back to its built-in default.
        config_file=config_file or config.config_file,
    )
