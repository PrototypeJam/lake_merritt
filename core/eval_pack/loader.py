import yaml
from pathlib import Path
from typing import List, Dict, Any, Union, Tuple
from core.eval_pack.schema import EvalPackV1
from core.registry import ComponentRegistry


class EvalPackLoader:
    """Loads and validates evaluation pack configurations."""
    
    def __init__(self):
        """Initialize the loader."""
        # Ensure built-in components are registered
        ComponentRegistry.discover_builtins()
    
    def load(self, source: Union[str, Path, Dict[str, Any]]) -> Tuple[EvalPackV1, List[str]]:
        """
        Load an evaluation pack from a YAML file or dictionary.
        
        Args:
            source: Either a file path (str or Path) to a YAML file, or a dictionary
                   containing the evaluation pack configuration.
        
        Returns:
            A tuple of (EvalPackV1 object, list of validation errors).
            If validation fails, the EvalPackV1 object may still be returned
            but should not be used for evaluation.
        """
        validation_errors = []
        
        # Load the configuration
        if isinstance(source, (str, Path)):
            config_dict = self._load_from_file(source, validation_errors)
            if config_dict is None:
                # Return None with the file loading error
                return None, validation_errors
        elif isinstance(source, dict):
            config_dict = source
        else:
            validation_errors.append(
                f"Invalid source type: {type(source).__name__}. "
                "Expected file path (str or Path) or dictionary."
            )
            return None, validation_errors
        
        # Create the EvalPackV1 object
        try:
            eval_pack = EvalPackV1(**config_dict)
        except Exception as e:
            validation_errors.append(f"Failed to create EvalPackV1 object: {str(e)}")
            return None, validation_errors
        
        # Validate the eval pack
        self._validate_eval_pack(eval_pack, validation_errors)
        
        return eval_pack, validation_errors
    
    def _load_from_file(self, file_path: Union[str, Path], 
                       validation_errors: List[str]) -> Dict[str, Any]:
        """
        Load YAML configuration from a file.
        
        Args:
            file_path: Path to the YAML file.
            validation_errors: List to append errors to.
        
        Returns:
            Dictionary containing the configuration, or None if loading failed.
        """
        path = Path(file_path)
        
        # Check if file exists
        if not path.exists():
            validation_errors.append(f"File not found: {path}")
            return None
        
        # Check if it's a file
        if not path.is_file():
            validation_errors.append(f"Path is not a file: {path}")
            return None
        
        # Load the YAML file
        try:
            with open(path, 'r') as f:
                config_dict = yaml.safe_load(f)
                
            if config_dict is None:
                validation_errors.append(f"Empty YAML file: {path}")
                return None
                
            if not isinstance(config_dict, dict):
                validation_errors.append(
                    f"YAML file must contain a dictionary at the root level, "
                    f"got {type(config_dict).__name__}: {path}"
                )
                return None
                
            return config_dict
            
        except yaml.YAMLError as e:
            validation_errors.append(f"Invalid YAML in file {path}: {str(e)}")
            return None
        except Exception as e:
            validation_errors.append(f"Error reading file {path}: {str(e)}")
            return None
    
    def _validate_eval_pack(self, eval_pack: EvalPackV1, 
                           validation_errors: List[str]) -> None:
        """
        Validate that all referenced components exist in the ComponentRegistry.
        
        Args:
            eval_pack: The EvalPackV1 object to validate.
            validation_errors: List to append validation errors to.
        """
        # Validate ingester
        ingester_type = eval_pack.ingestion.type
        try:
            ComponentRegistry.get_ingester(ingester_type)
        except ValueError:
            validation_errors.append(
                f"Unknown ingester type: '{ingester_type}'. "
                f"Available ingesters: {list(ComponentRegistry._ingesters.keys())}"
            )
        
        # Validate scorers in pipeline
        for i, stage in enumerate(eval_pack.pipeline):
            scorer_name = stage.scorer
            try:
                ComponentRegistry.get_scorer(scorer_name)
            except ValueError:
                validation_errors.append(
                    f"Unknown scorer in pipeline stage {i} ('{stage.name}'): '{scorer_name}'. "
                    f"Available scorers: {list(ComponentRegistry._scorers.keys())}"
                )
            
            # Validate on_fail value
            if stage.on_fail not in ["continue", "stop"]:
                validation_errors.append(
                    f"Invalid 'on_fail' value in pipeline stage {i} ('{stage.name}'): "
                    f"'{stage.on_fail}'. Must be either 'continue' or 'stop'."
                )
        
        # Validate reporting format if reporting is specified
        if eval_pack.reporting:
            valid_formats = ["markdown", "html", "pdf"]
            if eval_pack.reporting.format not in valid_formats:
                validation_errors.append(
                    f"Invalid reporting format: '{eval_pack.reporting.format}'. "
                    f"Must be one of: {valid_formats}"
                )