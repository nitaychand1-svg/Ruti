"""[FIX-10] Model version control."""
from __future__ import annotations

import json
import os
import pickle
from datetime import datetime
from typing import Any, Dict

import hashlib


class ModelVersionControl:
    """Version control for trained models."""

    def __init__(self, storage_path: str = "./models"):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)

    def save_model(
        self,
        components: Any,
        filename: str,
        X_sample: Any = None,
        config: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Save model with metadata."""
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.storage_path, filename)

        with open(filepath, "wb") as handle:
            pickle.dump(components, handle)

        metadata = {
            "version": version,
            "git_hash": self._get_git_hash(),
            "file_hash": self._file_hash(filepath),
            "created_at": datetime.now().isoformat(),
            "config": config or {},
            "n_features": int(X_sample.shape[1]) if X_sample is not None else None,
        }

        meta_path = filepath.replace(".pt", "_metadata.json")
        with open(meta_path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

        return metadata

    def load_model(self, filename: str) -> Dict[str, Any]:
        """Load model with validation."""
        filepath = os.path.join(self.storage_path, filename)
        meta_path = filepath.replace(".pt", "_metadata.json")

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        with open(meta_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)

        current_hash = self._file_hash(filepath)
        if current_hash != metadata.get("file_hash"):
            raise ValueError("Model file corrupted: hash mismatch")

        with open(filepath, "rb") as handle:
            components = pickle.load(handle)

        return {"components": components, "metadata": metadata}

    @staticmethod
    def _get_git_hash() -> str:
        try:
            import git  # type: ignore

            repo = git.Repo(search_parent_directories=True)
            return repo.head.object.hexsha[:8]
        except Exception:  # pragma: no cover
            return "unknown"

    @staticmethod
    def _file_hash(filepath: str) -> str:
        hasher = hashlib.sha256()
        with open(filepath, "rb") as handle:
            for chunk in iter(lambda: handle.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
