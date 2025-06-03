"""Model persistence utilities for saving and loading trained models."""

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .exceptions import ChurnRiskScorerError
from .logging_config import get_logger

logger = get_logger(__name__)


class ModelPersistenceError(ChurnRiskScorerError):
    """Exception raised for model persistence errors."""
    pass


class ModelPersistence:
    """Handles saving and loading of trained churn prediction models."""
    
    MODEL_FILE = "model.pkl"
    METADATA_FILE = "metadata.json"
    
    def __init__(self, base_path: str = "models"):
        """Initialize model persistence.
        
        Args:
            base_path: Base directory for storing models.
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Model persistence initialized with base path: {self.base_path}")
    
    def save_model(
        self,
        model: Any,
        model_name: str,
        feature_names: list,
        metrics: Optional[Dict[str, float]] = None,
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """Save a trained model with metadata.
        
        Args:
            model: The trained model object.
            model_name: Name identifier for the model.
            feature_names: List of feature names used for training.
            metrics: Optional dictionary of model performance metrics.
            extra_metadata: Optional additional metadata to store.
            
        Returns:
            Path to the saved model directory.
            
        Raises:
            ModelPersistenceError: If saving fails.
        """
        try:
            # Create timestamped model directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_dir = self.base_path / f"{model_name}_{timestamp}"
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Save the model
            model_path = model_dir / self.MODEL_FILE
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            logger.info(f"Model saved to {model_path}")
            
            # Prepare metadata
            metadata = {
                "model_name": model_name,
                "timestamp": timestamp,
                "created_at": datetime.now().isoformat(),
                "feature_names": feature_names,
                "num_features": len(feature_names),
                "model_type": type(model).__name__,
            }
            
            if metrics:
                metadata["metrics"] = metrics
            
            if extra_metadata:
                metadata["extra"] = extra_metadata
            
            # Save metadata
            metadata_path = model_dir / self.METADATA_FILE
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2, default=str)
            logger.info(f"Metadata saved to {metadata_path}")
            
            # Update latest symlink
            self._update_latest_link(model_name, model_dir)
            
            return model_dir
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise ModelPersistenceError(f"Failed to save model: {e}") from e
    
    def load_model(
        self,
        model_name: str,
        version: Optional[str] = None
    ) -> tuple:
        """Load a saved model and its metadata.
        
        Args:
            model_name: Name identifier for the model.
            version: Optional specific version timestamp. If None, loads latest.
            
        Returns:
            Tuple of (model, metadata).
            
        Raises:
            ModelPersistenceError: If loading fails.
        """
        try:
            if version:
                model_dir = self.base_path / f"{model_name}_{version}"
            else:
                # Try to load latest
                model_dir = self._find_latest_model(model_name)
            
            if not model_dir or not model_dir.exists():
                raise ModelPersistenceError(
                    f"Model '{model_name}' not found"
                )
            
            # Load model
            model_path = model_dir / self.MODEL_FILE
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            logger.info(f"Model loaded from {model_path}")
            
            # Load metadata
            metadata_path = model_dir / self.METADATA_FILE
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            logger.info(f"Metadata loaded from {metadata_path}")
            
            return model, metadata
            
        except ModelPersistenceError:
            raise
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise ModelPersistenceError(f"Failed to load model: {e}") from e
    
    def list_models(self, model_name: Optional[str] = None) -> list:
        """List available saved models.
        
        Args:
            model_name: Optional filter by model name.
            
        Returns:
            List of dictionaries with model information.
        """
        models = []
        
        for path in self.base_path.iterdir():
            if not path.is_dir() or path.name.startswith("."):
                continue
            
            # Skip symlinks
            if path.is_symlink():
                continue
            
            if model_name and not path.name.startswith(model_name):
                continue
            
            metadata_path = path / self.METADATA_FILE
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                models.append({
                    "path": str(path),
                    "name": metadata.get("model_name"),
                    "timestamp": metadata.get("timestamp"),
                    "created_at": metadata.get("created_at"),
                    "metrics": metadata.get("metrics", {})
                })
        
        # Sort by timestamp descending
        models.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return models
    
    def delete_model(self, model_name: str, version: str) -> bool:
        """Delete a specific model version.
        
        Args:
            model_name: Name identifier for the model.
            version: Version timestamp to delete.
            
        Returns:
            True if deletion was successful.
        """
        import shutil
        
        model_dir = self.base_path / f"{model_name}_{version}"
        
        if not model_dir.exists():
            logger.warning(f"Model directory not found: {model_dir}")
            return False
        
        shutil.rmtree(model_dir)
        logger.info(f"Deleted model: {model_dir}")
        return True
    
    def _find_latest_model(self, model_name: str) -> Optional[Path]:
        """Find the latest version of a model.
        
        Args:
            model_name: Name identifier for the model.
            
        Returns:
            Path to the latest model directory, or None if not found.
        """
        # First check for latest symlink
        latest_link = self.base_path / f"{model_name}_latest"
        if latest_link.exists() and latest_link.is_symlink():
            return latest_link.resolve()
        
        # Otherwise find by timestamp
        matching_dirs = [
            d for d in self.base_path.iterdir()
            if d.is_dir() and d.name.startswith(f"{model_name}_")
            and not d.is_symlink()
        ]
        
        if not matching_dirs:
            return None
        
        # Sort by name (timestamp) and return latest
        matching_dirs.sort(key=lambda x: x.name, reverse=True)
        return matching_dirs[0]
    
    def _update_latest_link(self, model_name: str, model_dir: Path) -> None:
        """Update the 'latest' symlink to point to the newest model.
        
        Args:
            model_name: Name identifier for the model.
            model_dir: Path to the new model directory.
        """
        latest_link = self.base_path / f"{model_name}_latest"
        
        try:
            if latest_link.exists() or latest_link.is_symlink():
                latest_link.unlink()
            latest_link.symlink_to(model_dir.name)
            logger.debug(f"Updated latest link: {latest_link} -> {model_dir.name}")
        except OSError as e:
            # Symlinks may not work on all systems
            logger.debug(f"Could not create symlink: {e}")
