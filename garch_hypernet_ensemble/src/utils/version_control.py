"""[FIX-10] Model version control."""
import os
import json
import hashlib
import pickle
from datetime import datetime
from typing import Any, Dict

class ModelVersionControl:
    """Version control for trained models."""
    
    def __init__(self, storage_path: str = "./models"):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
    
    def save_model(self, components: Any, filename: str,
                  X_sample: Any = None, config: Dict = None) -> Dict[str, Any]:
        """Save model with metadata."""
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.storage_path, filename)
        
        # Save components
        with open(filepath, 'wb') as f:
            pickle.dump(components, f)
        
        # Generate metadata
        metadata = {
            'version': version,
            'git_hash': self._get_git_hash(),
            'file_hash': self._file_hash(filepath),
            'created_at': datetime.now().isoformat(),
            'config': config or {},
            'n_features': X_sample.shape[1] if X_sample is not None else None
        }
        
        # Save metadata
        meta_path = filepath.replace('.pt', '_metadata.json')
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return metadata
    
    def load_model(self, filename: str) -> Dict[str, Any]:
        """Load model with validation."""
        filepath = os.path.join(self.storage_path, filename)
        meta_path = filepath.replace('.pt', '_metadata.json')
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        # Load metadata
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
        
        # Verify integrity
        current_hash = self._file_hash(filepath)
        if current_hash != metadata['file_hash']:
            raise ValueError("Model file corrupted: hash mismatch")
        
        # Load components
        with open(filepath, 'rb') as f:
            components = pickle.load(f)
        
        return {
            'components': components,
            'metadata': metadata
        }
    
    @staticmethod
    def _get_git_hash() -> str:
        """Get current git commit hash."""
        try:
            import git
            repo = git.Repo(search_parent_directories=True)
            return repo.head.object.hexsha[:8]
        except Exception:
            return "unknown"
    
    @staticmethod
    def _file_hash(filepath: str) -> str:
        """Calculate SHA256 hash of file."""
        hasher = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
