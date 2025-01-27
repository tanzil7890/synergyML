"""Model management system for SynergyML."""

import os
import json
import hashlib
import datetime
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import torch
import mlflow
from dataclasses import dataclass
from .utils import ModelCache

@dataclass
class ModelMetadata:
    """Model metadata information."""
    name: str
    version: str
    task: str
    framework: str
    input_shape: List[int]
    output_shape: List[int]
    performance_metrics: Dict[str, float]
    training_config: Dict[str, Any]
    dependencies: Dict[str, str]
    created_at: str
    updated_at: str
    hash: str

class ModelRegistry:
    """Central registry for model management."""
    
    def __init__(
        self,
        registry_path: str,
        mlflow_uri: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """Initialize model registry.
        
        Parameters
        ----------
        registry_path : str
            Path to model registry directory
        mlflow_uri : Optional[str]
            MLflow tracking URI
        cache_dir : Optional[str]
            Cache directory for model artifacts
        """
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        if mlflow_uri:
            mlflow.set_tracking_uri(mlflow_uri)
        
        self.cache = ModelCache(cache_dir) if cache_dir else None
        self._load_registry()
    
    def _load_registry(self):
        """Load registry information."""
        self.registry_file = self.registry_path / "registry.json"
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                self.registry = json.load(f)
        else:
            self.registry = {
                'models': {},
                'versions': {},
                'deployments': {}
            }
    
    def _save_registry(self):
        """Save registry information."""
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def _compute_model_hash(self, model_path: str) -> str:
        """Compute model file hash."""
        sha256_hash = hashlib.sha256()
        with open(model_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def register_model(
        self,
        name: str,
        model_path: str,
        task: str,
        framework: str,
        version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ModelMetadata:
        """Register a new model or version.
        
        Parameters
        ----------
        name : str
            Model name
        model_path : str
            Path to model file
        task : str
            Model task (e.g., 'scene_detection', 'sync')
        framework : str
            Model framework (e.g., 'pytorch', 'tensorflow')
        version : Optional[str]
            Model version (auto-generated if None)
        metadata : Optional[Dict[str, Any]]
            Additional model metadata
            
        Returns
        -------
        ModelMetadata
            Registered model metadata
        """
        # Load model to extract information
        model = torch.load(model_path)
        
        # Generate version if not provided
        if version is None:
            existing_versions = self.registry['versions'].get(name, [])
            version = f"v{len(existing_versions) + 1}"
        
        # Compute model hash
        model_hash = self._compute_model_hash(model_path)
        
        # Create metadata
        now = datetime.datetime.now().isoformat()
        model_metadata = ModelMetadata(
            name=name,
            version=version,
            task=task,
            framework=framework,
            input_shape=metadata.get('input_shape', []),
            output_shape=metadata.get('output_shape', []),
            performance_metrics=metadata.get('performance_metrics', {}),
            training_config=metadata.get('training_config', {}),
            dependencies=metadata.get('dependencies', {}),
            created_at=now,
            updated_at=now,
            hash=model_hash
        )
        
        # Update registry
        if name not in self.registry['models']:
            self.registry['models'][name] = {
                'task': task,
                'framework': framework,
                'latest_version': version
            }
        
        if name not in self.registry['versions']:
            self.registry['versions'][name] = []
        
        self.registry['versions'][name].append({
            'version': version,
            'metadata': model_metadata.__dict__,
            'path': str(model_path)
        })
        
        # Save to MLflow if configured
        if mlflow.active_run():
            mlflow.log_artifact(model_path, f"models/{name}/{version}")
            mlflow.log_params(metadata.get('training_config', {}))
            mlflow.log_metrics(metadata.get('performance_metrics', {}))
        
        self._save_registry()
        return model_metadata
    
    def get_model(
        self,
        name: str,
        version: Optional[str] = None,
        use_cache: bool = True
    ) -> torch.nn.Module:
        """Get a model by name and version.
        
        Parameters
        ----------
        name : str
            Model name
        version : Optional[str]
            Model version (latest if None)
        use_cache : bool
            Whether to use cache
            
        Returns
        -------
        torch.nn.Module
            Loaded model
        """
        if name not in self.registry['models']:
            raise ValueError(f"Model {name} not found in registry")
        
        # Get version info
        if version is None:
            version = self.registry['models'][name]['latest_version']
        
        version_info = None
        for v in self.registry['versions'][name]:
            if v['version'] == version:
                version_info = v
                break
        
        if version_info is None:
            raise ValueError(f"Version {version} not found for model {name}")
        
        # Check cache
        if use_cache and self.cache:
            cache_key = f"{name}_{version}_{version_info['metadata']['hash']}"
            cached_model = self.cache.load(cache_key)
            if cached_model is not None:
                return cached_model
        
        # Load model
        model = torch.load(version_info['path'])
        
        # Cache if enabled
        if use_cache and self.cache:
            self.cache.save(cache_key, model)
        
        return model
    
    def deploy_model(
        self,
        name: str,
        version: str,
        deployment_name: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Deploy a model version.
        
        Parameters
        ----------
        name : str
            Model name
        version : str
            Model version
        deployment_name : str
            Deployment name
        config : Dict[str, Any]
            Deployment configuration
            
        Returns
        -------
        Dict[str, Any]
            Deployment information
        """
        if name not in self.registry['models']:
            raise ValueError(f"Model {name} not found in registry")
        
        # Create deployment record
        deployment = {
            'model_name': name,
            'model_version': version,
            'config': config,
            'status': 'active',
            'deployed_at': datetime.datetime.now().isoformat()
        }
        
        self.registry['deployments'][deployment_name] = deployment
        self._save_registry()
        
        return deployment
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all registered models.
        
        Returns
        -------
        List[Dict[str, Any]]
            List of model information
        """
        models = []
        for name, info in self.registry['models'].items():
            versions = self.registry['versions'][name]
            models.append({
                'name': name,
                'task': info['task'],
                'framework': info['framework'],
                'latest_version': info['latest_version'],
                'versions': [v['version'] for v in versions]
            })
        return models
    
    def get_model_info(
        self,
        name: str,
        version: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get detailed model information.
        
        Parameters
        ----------
        name : str
            Model name
        version : Optional[str]
            Model version (latest if None)
            
        Returns
        -------
        Dict[str, Any]
            Model information
        """
        if name not in self.registry['models']:
            raise ValueError(f"Model {name} not found in registry")
        
        if version is None:
            version = self.registry['models'][name]['latest_version']
        
        for v in self.registry['versions'][name]:
            if v['version'] == version:
                return {
                    'name': name,
                    'version': version,
                    'metadata': v['metadata'],
                    'path': v['path']
                }
        
        raise ValueError(f"Version {version} not found for model {name}")
    
    def delete_model(
        self,
        name: str,
        version: Optional[str] = None,
        delete_files: bool = False
    ) -> None:
        """Delete a model or specific version.
        
        Parameters
        ----------
        name : str
            Model name
        version : Optional[str]
            Model version (all versions if None)
        delete_files : bool
            Whether to delete model files
        """
        if name not in self.registry['models']:
            raise ValueError(f"Model {name} not found in registry")
        
        if version is None:
            # Delete all versions
            if delete_files:
                for v in self.registry['versions'][name]:
                    if os.path.exists(v['path']):
                        os.remove(v['path'])
            
            del self.registry['models'][name]
            del self.registry['versions'][name]
        else:
            # Delete specific version
            versions = self.registry['versions'][name]
            for i, v in enumerate(versions):
                if v['version'] == version:
                    if delete_files and os.path.exists(v['path']):
                        os.remove(v['path'])
                    versions.pop(i)
                    break
            
            if not versions:
                del self.registry['models'][name]
                del self.registry['versions'][name]
            else:
                self.registry['versions'][name] = versions
                self.registry['models'][name]['latest_version'] = versions[-1]['version']
        
        self._save_registry() 