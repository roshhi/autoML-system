"""
Module 4: Preprocessing

This module executes all approved preprocessing steps from Module 3,
transforming the dataset into a clean, model-ready format.
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Tuple, Optional
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


def drop_columns(df: pd.DataFrame, approvals: Dict) -> Tuple[pd.DataFrame, List[str]]:
    """
    Drop approved columns.
    
    Args:
        df: DataFrame
        approvals: User approvals from Module 3
        
    Returns:
        DataFrame with columns dropped, list of dropped column names
    """
    columns_to_drop = []
    
    for key, approval in approvals.items():
        if approval.get('action') == 'drop' and approval.get('method') == 'column':
            col = approval.get('column')
            if col and col in df.columns:
                columns_to_drop.append(col)
    
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)
    
    return df, columns_to_drop


def apply_missing_value_fixes(df: pd.DataFrame, approvals: Dict) -> Tuple[pd.DataFrame, List[str]]:
    """
    Apply approved missing value imputation strategies.
    
    Args:
        df: DataFrame
        approvals: User approvals from Module 3
        
    Returns:
        DataFrame with imputed values, list of imputation logs
    """
    logs = []
    
    # Group imputations by method
    simple_impute = {'mean': [], 'median': [], 'mode': [], 'constant': []}
    advanced_impute = {'knn': [], 'iterative': []}
    
    for key, approval in approvals.items():
        if approval.get('action') == 'impute':
            col = approval.get('column')
            method = approval.get('method')
            
            if col and col in df.columns:
                if method in simple_impute:
                    simple_impute[method].append(col)
                elif method in advanced_impute:
                    advanced_impute[method].append(col)
    
    # Apply simple imputation
    for method, columns in simple_impute.items():
        if columns:
            if method == 'mean':
                for col in columns:
                    df[col].fillna(df[col].mean(), inplace=True)
                    logs.append(f"Imputed {col} with mean: {df[col].mean():.2f}")
            
            elif method == 'median':
                for col in columns:
                    df[col].fillna(df[col].median(), inplace=True)
                    logs.append(f"Imputed {col} with median: {df[col].median():.2f}")
            
            elif method == 'mode':
                for col in columns:
                    mode_val = df[col].mode()[0] if not df[col].mode().empty else 0
                    df[col].fillna(mode_val, inplace=True)
                    logs.append(f"Imputed {col} with mode: {mode_val}")
            
            elif method == 'constant':
                for col in columns:
                    fill_value = 0 if pd.api.types.is_numeric_dtype(df[col]) else 'Unknown'
                    df[col].fillna(fill_value, inplace=True)
                    logs.append(f"Imputed {col} with constant: {fill_value}")
    
    # Apply KNN imputation
    if advanced_impute['knn']:
        knn_cols = advanced_impute['knn']
        # Get only numeric columns for KNN
        numeric_cols = [col for col in knn_cols if pd.api.types.is_numeric_dtype(df[col])]
        
        if numeric_cols:
            imputer = KNNImputer(n_neighbors=5)
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
            logs.append(f"Applied KNN imputation (k=5) to {len(numeric_cols)} columns: {', '.join(numeric_cols)}")
    
    # Apply Iterative imputation
    if advanced_impute['iterative']:
        iter_cols = advanced_impute['iterative']
        numeric_cols = [col for col in iter_cols if pd.api.types.is_numeric_dtype(df[col])]
        
        if numeric_cols:
            imputer = IterativeImputer(max_iter=10, random_state=42)
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
            logs.append(f"Applied Iterative (MICE) imputation to {len(numeric_cols)} columns: {', '.join(numeric_cols)}")
    
    return df, logs


def apply_outlier_fixes(df: pd.DataFrame, approvals: Dict, eda_results: Dict) -> Tuple[pd.DataFrame, List[str]]:
    """
    Handle outliers based on approved strategy.
    
    Args:
        df: DataFrame
        approvals: User approvals from Module 3
        eda_results: EDA results containing IQR bounds
        
    Returns:
        DataFrame with outliers handled, list of logs
    """
    logs = []
    outlier_results = eda_results.get('outlier_results_iqr', {})
    
    for key, approval in approvals.items():
        if approval.get('issue_type') == 'outliers':
            col = approval.get('column')
            action = approval.get('action')
            
            if col and col in df.columns and col in outlier_results:
                bounds = outlier_results[col]
                lower_bound = bounds['lower_bound']
                upper_bound = bounds['upper_bound']
                
                if action == 'cap':
                    # Winsorization - cap at boundaries
                    original_outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                    logs.append(f"Capped {original_outliers} outliers in {col} at [{lower_bound:.2f}, {upper_bound:.2f}]")
                
                elif action == 'remove':
                    # Remove rows with outliers
                    original_len = len(df)
                    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                    removed = original_len - len(df)
                    logs.append(f"Removed {removed} rows with outliers in {col}")
                
                elif action == 'keep':
                    logs.append(f"Kept outliers in {col} as-is")
    
    return df, logs


def encode_categorical_features(df: pd.DataFrame, target_column: str, 
                                approvals: Dict, categorical_cols: List[str]) -> Tuple[pd.DataFrame, Dict, List[str]]:
    """
    Encode categorical features based on strategy.
    
    Args:
        df: DataFrame
        target_column: Target column name
        approvals: User approvals
        categorical_cols: List of categorical column names
        
    Returns:
        Encoded DataFrame, encoder mappings, logs
    """
    from sklearn.preprocessing import LabelEncoder
    
    logs = []
    encoders = {}
    
    # FIRST: Encode target column if it's categorical
    target_encoder = None
    if target_column in df.columns and not pd.api.types.is_numeric_dtype(df[target_column]):
        target_encoder = LabelEncoder()
        df[target_column] = target_encoder.fit_transform(df[target_column])
        encoders['target'] = {
            'method': 'label', 
            'encoder': target_encoder,
            'classes': target_encoder.classes_.tolist()
        }
        logs.append(f"Label encoded target '{target_column}': {dict(enumerate(target_encoder.classes_))}")
    
    # Exclude target from feature encoding
    features_to_encode = [col for col in categorical_cols if col != target_column and col in df.columns]
    
    for col in features_to_encode:
        # Check if specific encoding was requested in approvals
        encoding_method = 'onehot'  # default
        
        for key, approval in approvals.items():
            if approval.get('column') == col and approval.get('issue_type') == 'high_cardinality':
                if approval.get('action') == 'encode':
                    encoding_method = approval.get('method', 'onehot')
                break
        
        if encoding_method == 'onehot':
            # One-hot encoding
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
            encoders[col] = {'method': 'onehot', 'columns': dummies.columns.tolist()}
            logs.append(f"One-hot encoded {col}  {len(dummies.columns)} columns")
        
        elif encoding_method == 'target':
            # Target encoding - mean of target per category (now works because target is numeric)
            means = df.groupby(col)[target_column].mean()
            df[f'{col}_encoded'] = df[col].map(means)
            df = df.drop(columns=[col])
            encoders[col] = {'method': 'target', 'mapping': means.to_dict()}
            logs.append(f"Target encoded {col}  {col}_encoded")
        
        elif encoding_method == 'frequency':
            # Frequency encoding
            freq = df[col].value_counts(normalize=True)
            df[f'{col}_freq'] = df[col].map(freq)
            df = df.drop(columns=[col])
            encoders[col] = {'method': 'frequency', 'mapping': freq.to_dict()}
            logs.append(f"Frequency encoded {col}  {col}_freq")
    
    return df, encoders, logs


def scale_numerical_features(X_train: pd.DataFrame, X_test: pd.DataFrame,
                             method: str = 'standard') -> Tuple[pd.DataFrame, pd.DataFrame, object, List[str]]:
    """
    Scale numerical features.
    
    Args:
        X_train: Training features
        X_test: Test features
        method: 'standard', 'minmax', or 'robust'
        
    Returns:
        Scaled X_train, X_test, fitted scaler, logs
    """
    logs = []
    
    # Select scaler
    if method == 'standard':
        scaler = StandardScaler()
        scaler_name = "Standard Scaler (mean=0, std=1)"
    elif method == 'minmax':
        scaler = MinMaxScaler()
        scaler_name = "Min-Max Scaler [0, 1]"
    elif method == 'robust':
        scaler = RobustScaler()
        scaler_name = "Robust Scaler (median, IQR)"
    else:
        scaler = StandardScaler()
        scaler_name = "Standard Scaler (mean=0, std=1)"
    
    # Get numerical columns
    numerical_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    
    if numerical_cols:
        # Fit on train, transform both
        X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
        X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
        
        logs.append(f"Applied {scaler_name} to {len(numerical_cols)} numerical features")
    else:
        logs.append("No numerical features to scale")
    
    return X_train, X_test, scaler, logs


def apply_class_balancing(X_train: pd.DataFrame, y_train: pd.Series,
                          approvals: Dict) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Apply SMOTE if approved.
    
    Args:
        X_train: Training features
        y_train: Training target
        approvals: User approvals
        
    Returns:
        Balanced X_train, y_train, logs
    """
    logs = []
    
    # Check if SMOTE was approved
    apply_smote = False
    for key, approval in approvals.items():
        if approval.get('issue_type') == 'class_imbalance':
            apply_smote = approval.get('apply_smote', False)
            break
    
    if apply_smote:
        # Check minimum samples requirement
        class_counts = y_train.value_counts()
        min_samples = class_counts.min()
        
        if min_samples >= 6:  # SMOTE requires at least 6 samples
            original_len = len(X_train)
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            new_len = len(X_train)
            
            logs.append(f"Applied SMOTE: {original_len}  {new_len} samples")
            logs.append(f"Class distribution after SMOTE: {y_train.value_counts().to_dict()}")
        else:
            logs.append(f" Skipped SMOTE - minority class has only {min_samples} samples (need 6)")
    else:
        logs.append("SMOTE not applied (not approved)")
    
    return X_train, y_train, logs


def split_train_test(df: pd.DataFrame, target_column: str, approvals: Dict,
                     test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """
    Split data into train/test sets.
    
    Args:
        df: DataFrame
        target_column: Target column name
        approvals: User approvals
        test_size: Test set proportion
        
    Returns:
        X_train, X_test, y_train, y_test, logs
    """
    logs = []
    
    # Check if stratified split was approved
    stratified = False
    for key, approval in approvals.items():
        if approval.get('issue_type') == 'class_imbalance':
            stratified = approval.get('stratified_split', False)
            break
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Split
    if stratified:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        logs.append(f"Stratified train/test split: {len(X_train)} train, {len(X_test)} test")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        logs.append(f"Random train/test split: {len(X_train)} train, {len(X_test)} test")
    
    return X_train, X_test, y_train, y_test, logs


def execute_preprocessing_pipeline(df: pd.DataFrame, target_column: str, 
                                   approvals: Dict, eda_results: Dict,
                                   scaling_method: str = 'standard',
                                   test_size: float = 0.2) -> Tuple[Dict, List[str]]:
    """
    Execute complete preprocessing pipeline.
    
    Pipeline order:
    1. Drop columns
    2. Impute missing values
    3. Handle outliers
    4. Encode categorical features
    5. Split train/test
    6. Scale features (fit on train)
    7. Apply SMOTE (train only)
    
    Args:
        df: DataFrame
        target_column: Target column name
        approvals: User approvals from Module 3
        eda_results: EDA results
        scaling_method: Scaling method
        test_size: Test set proportion
        
    Returns:
        Dict with preprocessed data, list of all logs
    """
    all_logs = []
    df = df.copy()
    
    # Step 1: Drop columns
    st.write("**Step 1/7**: Dropping flagged columns...")
    df, dropped_cols = drop_columns(df, approvals)
    if dropped_cols:
        all_logs.append(f" Dropped {len(dropped_cols)} columns: {', '.join(dropped_cols)}")
    else:
        all_logs.append(" No columns to drop")
    
    # Step 2: Impute missing values
    st.write("**Step 2/7**: Imputing missing values...")
    df, impute_logs = apply_missing_value_fixes(df, approvals)
    all_logs.extend([f" {log}" for log in impute_logs])
    if not impute_logs:
        all_logs.append(" No missing values to impute")
    
    # Step 3: Handle outliers
    st.write("**Step 3/7**: Handling outliers...")
    df, outlier_logs = apply_outlier_fixes(df, approvals, eda_results)
    all_logs.extend([f" {log}" for log in outlier_logs])
    if not outlier_logs:
        all_logs.append(" No outliers to handle")
    
    # Step 4: Encode categorical features
    st.write("**Step 4/7**: Encoding categorical features...")
    categorical_features = eda_results.get('categorical_features', [])
    df, encoders, encode_logs = encode_categorical_features(df, target_column, approvals, categorical_features)
    all_logs.extend([f" {log}" for log in encode_logs])
    if not encode_logs:
        all_logs.append(" No categorical features to encode")
    
    # Step 5: Split train/test
    st.write("**Step 5/7**: Splitting train/test sets...")
    X_train, X_test, y_train, y_test, split_logs = split_train_test(df, target_column, approvals, test_size)
    all_logs.extend([f" {log}" for log in split_logs])
    
    # Step 6: Scale features
    st.write("**Step 6/7**: Scaling numerical features...")
    X_train, X_test, scaler, scale_logs = scale_numerical_features(X_train, X_test, scaling_method)
    all_logs.extend([f" {log}" for log in scale_logs])
    
    # Step 7: Apply SMOTE (train only)
    st.write("**Step 7/7**: Applying class balancing...")
    X_train, y_train, smote_logs = apply_class_balancing(X_train, y_train, approvals)
    all_logs.extend([f" {log}" for log in smote_logs])
    
    # Package results
    result = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'encoders': encoders,
        'dropped_columns': dropped_cols,
        'feature_names': X_train.columns.tolist()
    }
    
    return result, all_logs


def run_module_4(df: pd.DataFrame, target_column: str) -> None:
    """
    Main function for Module 4: Preprocessing.
    
    Args:
        df: DataFrame
        target_column: Target column name
    """
    st.markdown("""
        <div style="text-align: center; padding: 1.5rem 0;">
            <h1 style="font-size: 3rem; margin-bottom: 0.5rem;"> Data Preprocessing</h1>
            <p style="font-size: 1.15rem; color: #6b7280; font-weight: 400;">
                Apply approved fixes and prepare data for modeling
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Check dependencies
    if 'user_approvals' not in st.session_state or not st.session_state.user_approvals:
        st.error(" No preprocessing approvals found. Please complete Module 3 first.")
        if st.button(" Go to Issue Detection"):
            st.session_state.current_step = 3
            st.rerun()
        return
    
    if 'eda_results' not in st.session_state:
        st.error(" EDA results not found. Please complete Module 2 first.")
        if st.button(" Go to EDA"):
            st.session_state.current_step = 2
            st.rerun()
        return
    
    approvals = st.session_state.user_approvals
    eda_results = st.session_state.eda_results
    
    # Check if already completed
    preprocessing_completed = st.session_state.get('preprocessing_completed', False)
    
    if preprocessing_completed:
        st.info(" **Preprocessing already completed!** Displaying previous results. Click 'Re-run' to preprocess again.")
        
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button(" Re-run Preprocessing", use_container_width=True):
                st.session_state.preprocessing_completed = False
                st.session_state.preprocessed_data = None
                st.session_state.preprocessing_log = None
                st.rerun()
    
    # Show summary of approved fixes
    st.markdown("##  Approved Fixes Summary")
    
    # Count fixes by type
    drop_count = len([a for a in approvals.values() if a.get('action') == 'drop'])
    impute_count = len([a for a in approvals.values() if a.get('action') == 'impute'])
    outlier_count = len([a for a in approvals.values() if a.get('issue_type') == 'outliers'])
    balance_count = len([a for a in approvals.values() if a.get('issue_type') == 'class_imbalance'])
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Columns to Drop", drop_count)
    with col2:
        st.metric("Missing Value Fixes", impute_count)
    with col3:
        st.metric("Outlier Fixes", outlier_count)
    with col4:
        st.metric("Class Balance", "Yes" if balance_count > 0 else "No")
    
    st.markdown("---")
    
    # Configuration options
    if not preprocessing_completed:
        st.markdown("##  Preprocessing Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            scaling_method = st.radio(
                "**Scaling Method:**",
                options=['standard', 'minmax', 'robust'],
                format_func=lambda x: {
                    'standard': 'Standard Scaler (mean=0, std=1)',
                    'minmax': 'Min-Max Scaler [0, 1]',
                    'robust': 'Robust Scaler (median, IQR)'
                }[x],
                help="Scaling normalizes numerical features"
            )
        
        with col2:
            test_size = st.slider(
                "**Test Set Size:**",
                min_value=10,
                max_value=40,
                value=20,
                step=5,
                format="%d%%",
                help="Percentage of data for testing"
            ) / 100
        
        st.markdown("---")
        
        # Start preprocessing button
        if st.button(" Start Preprocessing", type="primary", use_container_width=True):
            st.markdown("##  Preprocessing Progress")
            
            progress_placeholder = st.empty()
            
            with st.spinner("Executing preprocessing pipeline..."):
                try:
                    result, logs = execute_preprocessing_pipeline(
                        df=df.copy(),
                        target_column=target_column,
                        approvals=approvals,
                        eda_results=eda_results,
                        scaling_method=scaling_method,
                        test_size=test_size
                    )
                    
                    # Store results
                    st.session_state.preprocessed_data = result
                    st.session_state.preprocessing_log = logs
                    st.session_state.preprocessing_completed = True
                    
                    st.success(" **Preprocessing Complete!**")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f" **Preprocessing Error**: {str(e)}")
                    st.exception(e)
    
    # Display results if completed
    if preprocessing_completed and st.session_state.get('preprocessed_data'):
        st.markdown("##  Preprocessing Results")
        
        result = st.session_state.preprocessed_data
        logs = st.session_state.preprocessing_log
        
        # Show logs
        with st.expander(" **Preprocessing Log**", expanded=True):
            for log in logs:
                st.markdown(f"- {log}")
        
        st.markdown("---")
        
        # Final dataset info
        st.markdown("##  Final Dataset")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("###  Training Set")
            st.metric("Samples", f"{len(result['X_train']):,}")
            st.metric("Features", len(result['feature_names']))
            st.caption(f"Class distribution: {result['y_train'].value_counts().to_dict()}")
        
        with col2:
            st.markdown("###  Test Set")
            st.metric("Samples", f"{len(result['X_test']):,}")
            st.metric("Features", len(result['feature_names']))
            st.caption(f"Class distribution: {result['y_test'].value_counts().to_dict()}")
        
        # Feature names
        with st.expander(" **Feature Names**"):
            st.write(result['feature_names'])
        
        # Show sample data
        with st.expander(" **Preview Training Data**"):
            preview_df = result['X_train'].head(10).copy()
            preview_df[target_column] = result['y_train'].head(10).values
            st.dataframe(preview_df, use_container_width=True)
        
        st.markdown("---")
        
        st.success("""
             **Data is ready for model training!**
            
            All preprocessing steps completed successfully. You can now proceed to train and evaluate classification models.
        """)
    
    # Navigation
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button(" Back to Issues", use_container_width=True):
            st.session_state.current_step = 3
            st.rerun()
    with col3:
        if preprocessing_completed:
            if st.button("Continue to Training ", use_container_width=True, type="primary"):
                st.session_state.current_step = 5
                st.rerun()
