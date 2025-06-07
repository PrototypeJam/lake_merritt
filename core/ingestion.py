"""
Data ingestion and validation utilities.
"""
import pandas as pd
import logging
from typing import List, Tuple, Union
from pathlib import Path
import json

from core.data_models import EvaluationItem, EvaluationMode

logger = logging.getLogger(__name__)


def validate_csv_columns(
    df: pd.DataFrame,
    required_columns: List[str],
) -> Tuple[bool, str]:
    """
    Validate that a DataFrame has the required columns.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
    
    Returns:
        Tuple of (is_valid, message)
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        return False, f"Missing required columns: {', '.join(missing_columns)}"
    
    # Check for empty required columns
    for col in required_columns:
        if df[col].isna().all():
            return False, f"Column '{col}' is empty"
    
    return True, "All required columns present and valid"


def load_evaluation_data(
    data: Union[pd.DataFrame, str, Path],
    mode: EvaluationMode = EvaluationMode.EVALUATE_EXISTING,
) -> List[EvaluationItem]:
    """
    Load evaluation data from various sources.
    
    Args:
        data: DataFrame, file path, or JSON string
        mode: Evaluation mode to determine required columns
    
    Returns:
        List of EvaluationItem objects
    """
    # Convert to DataFrame if needed
    if isinstance(data, (str, Path)):
        path = Path(data)
        if path.exists() and path.suffix == ".csv":
            df = pd.read_csv(path)
        elif path.exists() and path.suffix == ".json":
            with open(path) as f:
                json_data = json.load(f)
            df = pd.DataFrame(json_data)
        else:
            # Try to parse as JSON string
            try:
                json_data = json.loads(data)
                df = pd.DataFrame(json_data)
            except:
                raise ValueError(f"Unable to load data from: {data}")
    else:
        df = data
    
    # Validate columns based on mode
    required_columns = ["input", "expected_output"]
    if mode == EvaluationMode.EVALUATE_EXISTING:
        required_columns.append("output")
    
    is_valid, message = validate_csv_columns(df, required_columns)
    if not is_valid:
        raise ValueError(message)
    
    # Convert to EvaluationItem objects
    items = []
    for idx, row in df.iterrows():
        # Handle metadata columns
        metadata = {}
        for col in df.columns:
            if col not in ["id", "input", "output", "expected_output"]:
                value = row[col]
                # Convert numpy/pandas types to Python types
                if pd.notna(value):
                    if isinstance(value, (pd.Timestamp, pd.DatetimeTZDtype)):
                        metadata[col] = value.isoformat()
                    else:
                        metadata[col] = value
        
        item = EvaluationItem(
            id=str(row.get("id", idx + 1)),
            input=str(row["input"]),
            output=str(row.get("output", "")) if mode == EvaluationMode.EVALUATE_EXISTING else None,
            expected_output=str(row["expected_output"]),
            metadata=metadata,
        )
        items.append(item)
    
    logger.info(f"Loaded {len(items)} evaluation items")
    return items


def save_evaluation_data(
    items: List[EvaluationItem],
    output_path: Union[str, Path],
    format: str = "csv",
) -> None:
    """
    Save evaluation items to a file.
    
    Args:
        items: List of evaluation items
        output_path: Path to save the file
        format: Output format ('csv' or 'json')
    """
    output_path = Path(output_path)
    
    if format == "csv":
        # Convert to DataFrame
        data = []
        for item in items:
            row = {
                "id": item.id,
                "input": item.input,
                "output": item.output,
                "expected_output": item.expected_output,
            }
            # Add metadata columns
            row.update(item.metadata)
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        
    elif format == "json":
        # Convert to JSON
        data = [item.model_dump() for item in items]
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
    
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"Saved {len(items)} items to {output_path}")


def create_sample_data(
    num_items: int = 10,
    include_output: bool = True,
) -> pd.DataFrame:
    """
    Create sample evaluation data for testing.
    
    Args:
        num_items: Number of sample items to create
        include_output: Whether to include output column
    
    Returns:
        DataFrame with sample data
    """
    import random
    
    sample_prompts = [
        "What is the capital of France?",
        "Explain photosynthesis in simple terms.",
        "Write a haiku about programming.",
        "What are the primary colors?",
        "How do you make a peanut butter sandwich?",
        "What is 2 + 2?",
        "Translate 'hello' to Spanish.",
        "What is the largest planet in our solar system?",
        "Define artificial intelligence.",
        "What year did World War II end?",
    ]
    
    sample_outputs = [
        "The capital of France is Paris.",
        "Photosynthesis is how plants make food using sunlight, water, and carbon dioxide.",
        "Code flows like water\nBugs hide in the shadows deep\nDebugger finds all",
        "The primary colors are red, blue, and yellow (in traditional color theory).",
        "Spread peanut butter on one slice of bread, optionally add jam, then place another slice on top.",
        "2 + 2 equals 4.",
        "'Hello' in Spanish is 'Hola'.",
        "Jupiter is the largest planet in our solar system.",
        "AI is the simulation of human intelligence by machines, especially computer systems.",
        "World War II ended in 1945.",
    ]
    
    data = []
    for i in range(min(num_items, len(sample_prompts))):
        row = {
            "id": f"sample_{i+1}",
            "input": sample_prompts[i],
            "expected_output": sample_outputs[i],
        }
        
        if include_output:
            # Add some variation to outputs
            if random.random() > 0.7:
                # Introduce some errors
                row["output"] = sample_outputs[i].replace(".", "!")
            else:
                row["output"] = sample_outputs[i]
        
        data.append(row)
    
    # If we need more items, duplicate with variations
    while len(data) < num_items:
        base_item = random.choice(data[:len(sample_prompts)])
        new_item = base_item.copy()
        new_item["id"] = f"sample_{len(data)+1}"
        data.append(new_item)
    
    return pd.DataFrame(data)

