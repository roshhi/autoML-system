"""
Helper utility functions for AutoML Classification System
"""

import pandas as pd
import numpy as np
from typing import List, Tuple


def check_file_size(file, max_mb: int = 10) -> Tuple[bool, str]:
    """
    Validate uploaded file size.
    
    Args:
        file: Uploaded file object
        max_mb: Maximum allowed file size in MB
        
    Returns:
        Tuple of (is_valid, message)
    """
    if file is None:
        return False, "No file uploaded"
    
    file_size_mb = file.size / (1024 * 1024)
    
    if file_size_mb > max_mb:
        return False, f"File size ({file_size_mb:.2f} MB) exceeds maximum allowed size ({max_mb} MB)"
    
    return True, f"File size: {file_size_mb:.2f} MB"


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if division by zero.
    
    Args:
        a: Numerator
        b: Denominator
        default: Value to return if b is zero
        
    Returns:
        Result of division or default value
    """
    try:
        return a / b if b != 0 else default
    except:
        return default


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format a float as a percentage string.
    
    Args:
        value: Value between 0 and 1
        decimals: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{decimals}f}%"


def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    """
    Get list of numeric column names from dataframe.
    
    Args:
        df: Pandas DataFrame
        
    Returns:
        List of numeric column names
    """
    return df.select_dtypes(include=[np.number]).columns.tolist()


def get_categorical_columns(df: pd.DataFrame) -> List[str]:
    """
    Get list of categorical column names from dataframe.
    
    Args:
        df: Pandas DataFrame
        
    Returns:
        List of categorical column names
    """
    return df.select_dtypes(include=['object', 'category']).columns.tolist()


def clear_session_state(keys: List[str] = None):
    """
    Clear specified keys from Streamlit session state.
    
    Args:
        keys: List of keys to clear. If None, clears all keys.
    """
    import streamlit as st
    
    if keys is None:
        # Clear all session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
    else:
        # Clear specified keys
        for key in keys:
            if key in st.session_state:
                del st.session_state[key]


def validate_dataframe(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate that dataframe meets basic requirements.
    
    Args:
        df: Pandas DataFrame to validate
        
    Returns:
        Tuple of (is_valid, message)
    """
    if df is None:
        return False, "DataFrame is None"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df.columns) < 2:
        return False, "DataFrame must have at least 2 columns (features + target)"
    
    if len(df) < 10:
        return False, "DataFrame must have at least 10 rows for meaningful analysis"
    
    return True, "DataFrame is valid"


def validate_classification_dataset(df: pd.DataFrame, target_column: str) -> Tuple[bool, str]:
    """
    Comprehensive validation to determine if dataset is suitable for classification.
    Handles edge cases including numerically-encoded classes (1,2,3,4,5).
    
    Args:
        df: Pandas DataFrame
        target_column: Name of the target column
        
    Returns:
        Tuple of (is_valid, message)
    """
    # Validation thresholds
    THRESHOLDS = {
        'min_classes': 2,
        'max_classes': 50,
        'max_classes_hard': 100,
        'int_safe_unique_count': 20,
        'int_safe_ratio': 0.05,
        'int_ambiguous_ratio': 0.30,
        'float_safe_unique_count': 10,
        'float_safe_ratio': 0.02,
        'min_samples_for_ratio': 50,
    }
    
    # Step 1: Check if target column exists
    if target_column not in df.columns:
        return False, f" Column '{target_column}' not found in dataset"
    
    target_data = df[target_column]
    
    # Step 2: Check if target has any non-null values
    if target_data.isnull().all():
        return False, " Target column contains only missing values"
    
    # Step 3: Get unique values and counts (excluding nulls)
    target_clean = target_data.dropna()
    unique_values = target_clean.unique()
    num_unique = len(unique_values)
    total_count = len(target_clean)
    unique_ratio = num_unique / total_count if total_count > 0 else 0
    
    # Step 4: Check minimum classes
    if num_unique < THRESHOLDS['min_classes']:
        return False, (
            f" **Invalid target column**: only {num_unique} unique value(s) found.\n\n"
            f"Classification requires at least 2 distinct classes.\n\n"
            f" Check if you selected the correct target column."
        )
    
    # Step 5: Check maximum classes (hard limit)
    if num_unique > THRESHOLDS['max_classes_hard']:
        return False, (
            f" **Too many unique values**: {num_unique} classes detected.\n\n"
            f"This is extremely high for classification and likely indicates:\n"
            f"- Unique identifiers (IDs) instead of class labels\n"
            f"- Continuous regression values\n"
            f"- Wrong target column selected\n\n"
            f" Typical classification datasets have 2-50 classes.\n"
            f" Please select a different target column or verify your dataset."
        )
    
    # Step 6: Check data type - Categorical/String types are almost always classification
    if pd.api.types.is_categorical_dtype(target_data) or pd.api.types.is_object_dtype(target_data):
        # Categorical or string type
        if num_unique > THRESHOLDS['max_classes']:
            return False, (
                f" **High cardinality detected**: {num_unique} unique categorical values.\n\n"
                f"This is unusually high for classification. Please verify:\n"
                f"- Is this really a classification problem?\n"
                f"- Did you select the correct target column?\n"
                f"- Are these actual class labels or unique identifiers?\n\n"
                f" Typical classification datasets have 2-50 classes."
            )
        
        # Valid categorical classification
        return True, (
            f" **Valid classification dataset detected!**\n\n"
            f"- **Target Column**: {target_column}\n"
            f"- **Number of Classes**: {num_unique}\n"
            f"- **Data Type**: Categorical/String"
        )
    
    # Step 7: Boolean type
    if pd.api.types.is_bool_dtype(target_data):
        return True, (
            f" **Valid binary classification dataset!**\n\n"
            f"- **Target Column**: {target_column}\n"
            f"- **Number of Classes**: 2 (Boolean)\n"
            f"- **Data Type**: Boolean"
        )
    
    # Step 8: Numeric type - need more sophisticated checks
    if pd.api.types.is_numeric_dtype(target_data):
        
        # Check if all values are integers (no decimals)
        is_integer_like = target_clean.apply(lambda x: float(x).is_integer()).all()
        
        # INTEGER NUMERIC TARGETS
        if is_integer_like:
            # SPECIAL CHECK: Year detection (before general integer checks)
            # Detect sequential years like [2010, 2011, 2012, ..., 2020]
            if num_unique >= 15:  # At least 15 unique values
                sorted_values = sorted(unique_values)
                min_val = sorted_values[0]
                max_val = sorted_values[-1]
                
                # Check if values fall within typical year range (1800-2100)
                if 1800 <= min_val <= 2100 and 1800 <= max_val <= 2100:
                    # Check if values are mostly sequential/monotonic
                    diffs = np.diff(sorted_values)
                    
                    # Check 1: Small standard deviation of differences (mostly uniform gaps)
                    diff_std = np.std(diffs)
                    
                    # Check 2: Percentage of consecutive values (diff = 1)
                    consecutive_pct = (diffs == 1).sum() / len(diffs) if len(diffs) > 0 else 0
                    
                    # If mostly sequential (std < 2.0 OR >70% consecutive)
                    if diff_std < 2.0 or consecutive_pct > 0.70:
                        return False, (
                            f" **Year column detected (Time series data)**\n\n"
                            f"The target column '{target_column}' appears to contain year values:\n"
                            f"- **Range**: {int(min_val)} to {int(max_val)}\n"
                            f"- **Unique values**: {num_unique}\n"
                            f"- **Sequential pattern**: {consecutive_pct*100:.1f}% consecutive\n\n"
                            f"**Year columns are continuous time series, not classification targets.**\n\n"
                            f"Common examples:\n"
                            f"- Publication years (2010, 2011, 2012, ...)\n"
                            f"- Birth years (1980, 1985, 1990, ...)\n"
                            f"- Calendar years for time series data\n\n"
                            f"**This system only supports classification tasks** with discrete class labels.\n\n"
                            f" If you're analyzing time-based data, consider:\n"
                            f"- Using time series analysis tools\n"
                            f"- Extracting categorical features (e.g., decade, era)\n"
                            f"- Selecting a different target column"
                        )
            
            # Very low unique count - clearly classification
            if num_unique <= THRESHOLDS['int_safe_unique_count']:
                return True, (
                    f" **Valid classification dataset detected!**\n\n"
                    f"- **Target Column**: {target_column}\n"
                    f"- **Number of Classes**: {num_unique}\n"
                    f"- **Data Type**: Integer (Numerically-encoded classes)\n"
                    f"- **Unique Values**: {sorted(unique_values)[:10]}{'...' if num_unique > 10 else ''}"
                )
            
            # Very low ratio - likely encoded classes (e.g., 5 classes in 1000 samples)
            if unique_ratio <= THRESHOLDS['int_safe_ratio']:
                return True, (
                    f" **Valid classification dataset detected!**\n\n"
                    f"- **Target Column**: {target_column}\n"
                    f"- **Number of Classes**: {num_unique}\n"
                    f"- **Data Type**: Integer (Numerically-encoded classes)\n"
                    f"- **Unique Ratio**: {unique_ratio*100:.2f}% (low - indicates discrete classes)\n"
                    f"- **Sample Values**: {sorted(unique_values)[:10]}{'...' if num_unique > 10 else ''}"
                )
            
            # High ratio - likely regression or IDs
            if unique_ratio > THRESHOLDS['int_ambiguous_ratio']:
                return False, (
                    f" **This appears to be a regression dataset, not classification!**\n\n"
                    f"The target column '{target_column}' has:\n"
                    f"- **{num_unique} unique integer values** ({unique_ratio*100:.1f}% of all samples)\n"
                    f"- This high uniqueness suggests continuous or ID values, not discrete classes\n\n"
                    f"**Common examples of regression targets:**\n"
                    f"- Ages (18, 19, 20, ..., 80)\n"
                    f"- Prices (100, 150, 200, ...)\n"
                    f"- Years (2010, 2011, 2012, ...)\n"
                    f"- Customer IDs\n\n"
                    f"**This system only supports classification tasks** with discrete class labels.\n\n"
                    f" Please upload a classification dataset or select a different target column."
                )
            
            # Ambiguous zone - moderate ratio, could go either way
            # Between safe_ratio (5%) and ambiguous_ratio (30%)
            if num_unique <= THRESHOLDS['max_classes']:
                # Within acceptable class count, but high ratio - give benefit of doubt
                return True, (
                    f" **Classification dataset accepted with caution**\n\n"
                    f"- **Target Column**: {target_column}\n"
                    f"- **Number of Classes**: {num_unique}\n"
                    f"- **Data Type**: Integer\n"
                    f"- **Unique Ratio**: {unique_ratio*100:.1f}%\n\n"
                    f" **Warning**: The unique ratio is moderately high ({unique_ratio*100:.1f}%).\n"
                    f"If this is actually a regression problem (e.g., age, price, count), "
                    f"please use a regression tool instead.\n\n"
                    f" Proceeding as multi-class classification with {num_unique} classes."
                )
            else:
                # Too many classes
                return False, (
                    f" **Too many classes detected**: {num_unique} unique values\n\n"
                    f"Classification with more than {THRESHOLDS['max_classes']} classes is unusual.\n\n"
                    f" Please verify this is a classification problem."
                )
        
        # FLOAT NUMERIC TARGETS
        else:
            # Very low unique count for floats
            if num_unique <= THRESHOLDS['float_safe_unique_count']:
                return True, (
                    f" **Valid classification dataset detected!**\n\n"
                    f"- **Target Column**: {target_column}\n"
                    f"- **Number of Classes**: {num_unique}\n"
                    f"- **Data Type**: Float (discrete classes)\n"
                    f"- **Unique Values**: {sorted(unique_values)[:10]}"
                )
            
            # Very low ratio for floats (extremely rare but could be valid)
            if unique_ratio <= THRESHOLDS['float_safe_ratio']:
                return True, (
                    f" **Valid classification dataset detected!**\n\n"
                    f"- **Target Column**: {target_column}\n"
                    f"- **Number of Classes**: {num_unique}\n"
                    f"- **Data Type**: Float (discrete classes)\n"
                    f"- **Unique Ratio**: {unique_ratio*100:.2f}%"
                )
            
            # High uniqueness in floats - almost certainly regression
            return False, (
                f" **This is a regression dataset, not classification!**\n\n"
                f"The target column '{target_column}' contains:\n"
                f"- **{num_unique} unique float values** ({unique_ratio*100:.1f}% of samples)\n"
                f"- Float values with high uniqueness indicate continuous measurements\n\n"
                f"**Common examples:**\n"
                f"- Temperatures (36.5, 37.2, 38.1, ...)\n"
                f"- Prices ($19.99, $29.95, $45.50, ...)\n"
                f"- Measurements (height, weight, distance)\n"
                f"- Percentages (0.156, 0.892, 0.234, ...)\n\n"
                f"**This system only supports classification tasks** with discrete class labels.\n\n"
                f" Please upload a classification dataset with categorical targets."
            )
    
    # Step 9: Unknown type - reject with helpful message
    return False, (
        f" **Unsupported target data type**: {target_data.dtype}\n\n"
        f"Valid classification targets should be:\n"
        f"- Categorical/String (e.g., 'Yes/No', 'Cat A/B/C')\n"
        f"- Integer with low cardinality (e.g., 0/1, 1/2/3/4/5)\n"
        f"- Boolean (True/False)\n\n"
        f" Please select a different target column."
    )


def infer_target_column(df: pd.DataFrame) -> str:
    """
    Try to infer the target column from common naming patterns.
    
    Args:
        df: Pandas DataFrame
        
    Returns:
        Name of likely target column, or last column name as fallback
    """
    common_target_names = [
        'target', 'label', 'class', 'category', 'outcome', 
        'result', 'output', 'prediction', 'y'
    ]
    
    # Check if any column name matches common patterns (case-insensitive)
    for col in df.columns:
        if col.lower() in common_target_names:
            return col
    
    # Fallback: return last column
    return df.columns[-1]
