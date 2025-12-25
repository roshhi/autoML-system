"""
Module 3: Issue Detection & User Approval

This module analyzes EDA results to detect data quality issues and provides
an interactive interface for users to approve preprocessing fixes.
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Optional, Tuple


def detect_missing_value_issues(df: pd.DataFrame, missing_data: Dict, correlation_matrix: Optional[pd.DataFrame] = None) -> List[Dict]:
    """
    Detect and categorize missing value issues with intelligent imputation recommendations.
    
    Args:
        df: DataFrame
        missing_data: Missing value analysis from Module 2
        correlation_matrix: Correlation matrix from EDA (optional)
        
    Returns:
        List of issue dictionaries
    """
    issues = []
    
    for column in missing_data['columns_with_missing']:
        missing_pct = missing_data['missing_percentages'][column]
        missing_count = missing_data['missing_counts'][column]
        
        # Check if column has strong correlations with other features
        has_strong_correlations = False
        correlated_features = []
        
        if correlation_matrix is not None and column in correlation_matrix.columns:
            # Get correlations for this column (excluding self-correlation)
            correlations = correlation_matrix[column].drop(column, errors='ignore')
            # Find features with correlation > 0.5
            strong_corr = correlations[abs(correlations) > 0.5]
            if len(strong_corr) > 0:
                has_strong_correlations = True
                correlated_features = strong_corr.index.tolist()
        
        # Determine severity and recommendation based on missing % and correlations
        if missing_pct > 70:
            severity = 'critical'
            recommendation = {'action': 'drop', 'method': 'column'}
            reason = 'Too much missing data (>70%) to reliably impute'
        elif missing_pct > 30:
            severity = 'high'
            if has_strong_correlations:
                recommendation = {'action': 'impute', 'method': 'iterative'}
                reason = f'High missingness but {len(correlated_features)} correlated features found - iterative imputation recommended'
            else:
                recommendation = {'action': 'drop', 'method': 'column'}
                reason = 'High missingness (>30%) with no strong correlations - consider dropping'
        elif missing_pct > 10:
            severity = 'medium'
            if has_strong_correlations and pd.api.types.is_numeric_dtype(df[column]):
                recommendation = {'action': 'impute', 'method': 'knn'}
                reason = f'Moderate missingness with {len(correlated_features)} correlated features - KNN imputation recommended'
            elif pd.api.types.is_numeric_dtype(df[column]):
                recommendation = {'action': 'impute', 'method': 'median'}
                reason = 'Moderate missingness - median imputation recommended'
            else:
                recommendation = {'action': 'impute', 'method': 'mode'}
                reason = 'Moderate missingness - mode imputation recommended'
        else:
            severity = 'low'
            if has_strong_correlations and pd.api.types.is_numeric_dtype(df[column]):
                recommendation = {'action': 'impute', 'method': 'knn'}
                reason = f'Low missingness with correlations - KNN imputation for accuracy'
            elif pd.api.types.is_numeric_dtype(df[column]):
                recommendation = {'action': 'impute', 'method': 'mean'}
                reason = 'Low missingness - mean imputation safe'
            else:
                recommendation = {'action': 'impute', 'method': 'mode'}
                reason = 'Low missingness - mode imputation safe'
        
        # Build alternatives based on data type and correlations
        alternatives = []
        
        if pd.api.types.is_numeric_dtype(df[column]):
            # Advanced methods (if not already recommended)
            if has_strong_correlations:
                if recommendation['method'] != 'knn':
                    alternatives.append({
                        'action': 'impute', 
                        'method': 'knn', 
                        'label': f'KNN Imputation (uses {len(correlated_features)} correlated features)'
                    })
                if recommendation['method'] != 'iterative':
                    alternatives.append({
                        'action': 'impute', 
                        'method': 'iterative', 
                        'label': 'Iterative Imputation (MICE - model-based)'
                    })
            
            # Simple methods
            if recommendation['method'] != 'mean':
                alternatives.append({'action': 'impute', 'method': 'mean', 'label': 'Impute with Mean (simple)'})
            if recommendation['method'] != 'median':
                alternatives.append({'action': 'impute', 'method': 'median', 'label': 'Impute with Median (simple)'})
            
            alternatives.append({'action': 'impute', 'method': 'constant', 'label': 'Fill with 0'})
        else:
            # Categorical
            if recommendation['method'] != 'mode':
                alternatives.append({'action': 'impute', 'method': 'mode', 'label': 'Impute with Mode'})
            alternatives.append({'action': 'impute', 'method': 'constant', 'label': 'Fill with "Unknown"'})
        
        # Always offer drop option
        if recommendation['action'] != 'drop':
            alternatives.append({'action': 'drop', 'method': 'column', 'label': 'Drop Column'})
        
        issue = {
            'column': column,
            'issue_type': 'missing_values',
            'severity': severity,
            'description': f'{missing_pct:.1f}% missing values ({missing_count:,} rows)',
            'details': {
                'missing_count': int(missing_count),
                'missing_pct': missing_pct,
                'total_rows': len(df),
                'has_correlations': has_strong_correlations,
                'correlated_features': correlated_features[:3] if correlated_features else []  # Top 3
            },
            'recommendation': recommendation,
            'reason': reason,
            'alternatives': alternatives
        }
        
        issues.append(issue)
    
    return issues


def detect_outlier_issues(df: pd.DataFrame, outlier_results: Dict, threshold_pct: float = 5.0) -> List[Dict]:
    """
    Detect columns with significant outliers.
    
    Args:
        df: DataFrame
        outlier_results: Outlier detection results from Module 2 (IQR method)
        threshold_pct: Percentage threshold to flag as issue
        
    Returns:
        List of outlier issue dictionaries
    """
    issues = []
    
    for column, result in outlier_results.items():
        outlier_pct = result['outlier_percentage']
        outlier_count = result['outlier_count']
        
        if outlier_pct > threshold_pct:
            # Determine severity
            if outlier_pct > 20:
                severity = 'high'
            elif outlier_pct > 10:
                severity = 'medium'
            else:
                severity = 'low'
            
            issue = {
                'column': column,
                'issue_type': 'outliers',
                'severity': severity,
                'description': f'{outlier_pct:.1f}% outliers detected ({outlier_count:,} values)',
                'details': {
                    'outlier_count': outlier_count,
                    'outlier_pct': outlier_pct,
                    'lower_bound': result['lower_bound'],
                    'upper_bound': result['upper_bound']
                },
                'recommendation': {'action': 'cap', 'method': 'iqr'},
                'reason': 'Cap outliers at IQR boundaries (winsorization)',
                'alternatives': [
                    {'action': 'remove', 'method': 'rows', 'label': 'Remove Outlier Rows'},
                    {'action': 'keep', 'method': 'none', 'label': 'Keep As-Is'}
                ]
            }
            
            issues.append(issue)
    
    return issues


def detect_class_imbalance(df: pd.DataFrame, target_column: str) -> Optional[Dict]:
    """
    Check for class imbalance in target variable.
    
    Args:
        df: DataFrame
        target_column: Target column name
        
    Returns:
        Issue dictionary or None if balanced
    """
    value_counts = df[target_column].value_counts()
    
    if len(value_counts) < 2:
        return None  # Can't have imbalance with single class
    
    majority_count = value_counts.iloc[0]
    minority_count = value_counts.iloc[-1]
    imbalance_ratio = majority_count / minority_count
    
    # Only flag if ratio > 3:1
    if imbalance_ratio > 3.0:
        # Determine severity
        if imbalance_ratio > 10:
            severity = 'critical'
        elif imbalance_ratio > 5:
            severity = 'high'
        else:
            severity = 'medium'
        
        majority_pct = (majority_count / len(df)) * 100
        minority_pct = (minority_count / len(df)) * 100
        
        issue = {
            'column': target_column,
            'issue_type': 'class_imbalance',
            'severity': severity,
            'description': f'Imbalance ratio {imbalance_ratio:.1f}:1 (Majority: {majority_pct:.1f}% vs Minority: {minority_pct:.1f}%)',
            'details': {
                'imbalance_ratio': imbalance_ratio,
                'majority_class': value_counts.index[0],
                'majority_count': int(majority_count),
                'majority_pct': majority_pct,
                'minority_class': value_counts.index[-1],
                'minority_count': int(minority_count),
                'minority_pct': minority_pct
            },
            'recommendation': {
                'apply_smote': True,
                'stratified_split': True,
                'class_weights': True
            },
            'reason': 'Apply multiple techniques to handle imbalance',
            'alternatives': []
        }
        
        return issue
    
    return None


def detect_high_cardinality_features(df: pd.DataFrame, categorical_cols: List[str], threshold: int = 20) -> List[Dict]:
    """
    Detect categorical features with too many unique values.
    
    Args:
        df: DataFrame
        categorical_cols: List of categorical column names
        threshold: Unique value threshold
        
    Returns:
        List of high cardinality issues
    """
    issues = []
    
    for column in categorical_cols:
        unique_count = df[column].nunique()
        
        if unique_count > threshold:
            # Determine severity and recommendation
            if unique_count > 100:
                severity = 'high'
                recommendation = {'action': 'drop', 'method': 'column'}
                reason = 'Too many categories - likely not useful'
                alternatives = [
                    {'action': 'encode', 'method': 'target', 'label': 'Target Encoding'},
                    {'action': 'encode', 'method': 'frequency', 'label': 'Frequency Encoding'}
                ]
            elif unique_count > 50:
                severity = 'medium'
                recommendation = {'action': 'encode', 'method': 'target'}
                reason = 'High cardinality - target encoding recommended'
                alternatives = [
                    {'action': 'encode', 'method': 'frequency', 'label': 'Frequency Encoding'},
                    {'action': 'drop', 'method': 'column', 'label': 'Drop Column'},
                    {'action': 'keep', 'method': 'onehot', 'label': 'One-Hot (tree models)'}
                ]
            else:
                severity = 'low'
                recommendation = {'action': 'keep', 'method': 'onehot'}
                reason = 'Moderate cardinality - one-hot encoding acceptable'
                alternatives = [
                    {'action': 'encode', 'method': 'target', 'label': 'Target Encoding'},
                    {'action': 'encode', 'method': 'frequency', 'label': 'Frequency Encoding'}
                ]
            
            issue = {
                'column': column,
                'issue_type': 'high_cardinality',
                'severity': severity,
                'description': f'{unique_count} unique categories',
                'details': {
                    'unique_count': unique_count,
                    'sample_values': df[column].value_counts().head(5).to_dict()
                },
                'recommendation': recommendation,
                'reason': reason,
                'alternatives': alternatives
            }
            
            issues.append(issue)
    
    return issues


def detect_constant_features(df: pd.DataFrame, threshold: float = 0.95) -> List[Dict]:
    """
    Detect constant or near-constant features.
    
    Args:
        df: DataFrame
        threshold: Threshold for near-constant (e.g., 0.95 = 95% same value)
        
    Returns:
        List of constant feature issues
    """
    issues = []
    
    for column in df.columns:
        value_counts = df[column].value_counts()
        
        if len(value_counts) == 1:
            # Completely constant
            issue = {
                'column': column,
                'issue_type': 'constant_feature',
                'severity': 'medium',
                'description': 'Constant value - no variance',
                'details': {
                    'unique_count': 1,
                    'constant_value': value_counts.index[0]
                },
                'recommendation': {'action': 'drop', 'method': 'column'},
                'reason': 'No predictive power - all values identical',
                'alternatives': []
            }
            issues.append(issue)
        
        elif len(value_counts) > 1:
            # Check for near-constant
            most_common_pct = (value_counts.iloc[0] / len(df))
            
            if most_common_pct >= threshold:
                issue = {
                    'column': column,
                    'issue_type': 'constant_feature',
                    'severity': 'low',
                    'description': f'Near-constant - {most_common_pct*100:.1f}% same value',
                    'details': {
                        'unique_count': len(value_counts),
                        'most_common_value': value_counts.index[0],
                        'most_common_pct': most_common_pct * 100
                    },
                    'recommendation': {'action': 'drop', 'method': 'column'},
                    'reason': 'Very low variance - minimal predictive power',
                    'alternatives': [
                        {'action': 'keep', 'method': 'none', 'label': 'Keep Feature'}
                    ]
                }
                issues.append(issue)
    
    return issues


def analyze_all_issues(df: pd.DataFrame, target_column: str, eda_results: Dict) -> Dict:
    """
    Run all issue detection functions and categorize by severity.
    
    Args:
        df: DataFrame
        target_column: Target column name
        eda_results: Results from Module 2
        
    Returns:
        Dictionary categorizing all issues
    """
    all_issues = []
    
    # Get correlation matrix if available
    correlation_matrix = eda_results.get('correlation_matrix', None)
    
    # 1. Missing values (with correlation-aware imputation)
    if eda_results['missing_data']['num_affected_columns'] > 0:
        missing_issues = detect_missing_value_issues(
            df, 
            eda_results['missing_data'],
            correlation_matrix=correlation_matrix
        )
        all_issues.extend(missing_issues)
    
    # 2. Outliers
    if 'outlier_results_iqr' in eda_results and eda_results['outlier_results_iqr']:
        outlier_issues = detect_outlier_issues(df, eda_results['outlier_results_iqr'])
        all_issues.extend(outlier_issues)
    
    # 3. Class imbalance
    imbalance_issue = detect_class_imbalance(df, target_column)
    if imbalance_issue:
        all_issues.append(imbalance_issue)
    
    # 4. High cardinality (exclude target)
    categorical_features = [col for col in eda_results.get('categorical_features', []) if col != target_column]
    if categorical_features:
        cardinality_issues = detect_high_cardinality_features(df, categorical_features)
        all_issues.extend(cardinality_issues)
    
    # 5. Constant features (exclude target)
    constant_issues = detect_constant_features(df[[col for col in df.columns if col != target_column]])
    all_issues.extend(constant_issues)
    
    # Categorize by severity
    categorized = {
        'critical': [issue for issue in all_issues if issue['severity'] == 'critical'],
        'high': [issue for issue in all_issues if issue['severity'] == 'high'],
        'medium': [issue for issue in all_issues if issue['severity'] == 'medium'],
        'low': [issue for issue in all_issues if issue['severity'] == 'low'],
        'all': all_issues
    }
    
    return categorized


def display_issue_card(issue: Dict, issue_num: int, key_prefix: str) -> Optional[Dict]:
    """
    Display a single issue card with interactive approval.
    
    Args:
        issue: Issue dictionary
        issue_num: Issue number for display
        key_prefix: Unique key prefix for widgets
        
    Returns:
        User-selected fix or None
    """
    # Severity colors
    severity_colors = {
        'critical': ('#e74c3c', '#ff6b6b'),
        'high': ('#f39c12', '#ffc107'),
        'medium': ('#3498db', '#5dade2'),
        'low': ('#27ae60', '#2ecc71')
    }
    
    severity_emoji = {
        'critical': '',
        'high': '',
        'medium': '',
        'low': ''
    }
    
    color, light_color = severity_colors.get(issue['severity'], ('#95a5a6', '#bdc3c7'))
    emoji = severity_emoji.get(issue['severity'], '')
    
    # Issue type labels
    type_labels = {
        'missing_values': ' Missing Values',
        'outliers': ' Outliers',
        'class_imbalance': ' Class Imbalance',
        'high_cardinality': ' High Cardinality',
        'constant_feature': ' Constant Feature'
    }
    
    type_label = type_labels.get(issue['issue_type'], issue['issue_type'].replace('_', ' ').title())
    
    with st.expander(f"{emoji} **Issue #{issue_num}: {issue['column']}** - {type_label}", expanded=True):
        st.markdown(f"**{issue['description']}**")
        st.caption(f"Severity: {issue['severity'].upper()}")
        
        # Show correlated features if available (for missing values)
        if issue['issue_type'] == 'missing_values' and issue['details'].get('has_correlations'):
            corr_features = issue['details'].get('correlated_features', [])
            if corr_features:
                st.caption(f" Correlated with: {', '.join(corr_features)}")
        
        # Show recommendation
        st.info(f" **Recommendation**: {issue['reason']}")
        
        # Handle different issue types
        if issue['issue_type'] == 'class_imbalance':
            # Special handling for class imbalance
            st.markdown("**Suggested Actions:**")
            apply_smote = st.checkbox(
                "Apply SMOTE (Synthetic Oversampling)",
                value=issue['recommendation'].get('apply_smote', False),
                key=f"{key_prefix}_smote"
            )
            stratified = st.checkbox(
                "Use Stratified Train/Test Split",
                value=issue['recommendation'].get('stratified_split', True),
                key=f"{key_prefix}_stratified"
            )
            class_weights = st.checkbox(
                "Apply Class Weights in Models",
                value=issue['recommendation'].get('class_weights', True),
                key=f"{key_prefix}_weights"
            )
            
            return {
                'issue_type': issue['issue_type'],
                'column': issue['column'],
                'apply_smote': apply_smote,
                'stratified_split': stratified,
                'class_weights': class_weights
            }
        else:
            # Standard fix selection
            st.markdown("**Select Fix:**")
            
            # Build options
            rec = issue['recommendation']
            if rec['action'] == 'impute':
                rec_label = f"Impute with {rec['method'].title()}"
            elif rec['action'] == 'drop':
                rec_label = "Drop Column"
            elif rec['action'] == 'cap':
                rec_label = "Cap Outliers (Winsorization)"
            elif rec['action'] == 'encode':
                rec_label = f"{rec['method'].title()} Encoding"
            elif rec['action'] == 'keep':
                rec_label = f"Keep ({rec['method']})"
            else:
                rec_label = f"{rec['action'].title()} - {rec['method'].title()}"
            
            options = [rec_label] + [alt['label'] for alt in issue.get('alternatives', [])]
            
            selected = st.radio(
                "Choose action:",
                options,
                index=0,
                key=f"{key_prefix}_fix",
                label_visibility="collapsed"
            )
            
            # Parse selection
            if selected == rec_label:
                return {
                    'issue_type': issue['issue_type'],
                    'column': issue['column'],
                    'action': rec['action'],
                    'method': rec['method']
                }
            else:
                # Find matching alternative
                for alt in issue.get('alternatives', []):
                    if alt['label'] == selected:
                        return {
                            'issue_type': issue['issue_type'],
                            'column': issue['column'],
                            'action': alt['action'],
                            'method': alt['method']
                        }
    
    return None


def run_module_3(df: pd.DataFrame, target_column: str) -> None:
    """
    Main function for Module 3: Issue Detection & User Approval.
    
    Args:
        df: DataFrame
        target_column: Target column name
    """
    st.markdown("""
        <div style="text-align: center; padding: 1.5rem 0;">
            <h1 style="font-size: 3rem; margin-bottom: 0.5rem;"> Issue Detection & Approval</h1>
            <p style="font-size: 1.15rem; color: #6b7280; font-weight: 400;">
                Review data quality issues and approve preprocessing solutions
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Check if already completed
    issues_already_detected = (
        'detected_issues' in st.session_state and
        st.session_state.detected_issues is not None and
        'issues_completed' in st.session_state and
        st.session_state.issues_completed == True
    )
    
    if issues_already_detected:
        st.info(" **Issues already analyzed!** Displaying previous results. Click 'Re-analyze' to detect again.")
        
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button(" Re-analyze Issues", use_container_width=True):
                st.session_state.issues_completed = False
                st.session_state.detected_issues = None
                st.session_state.user_approvals = None
                st.rerun()
    
    # Get EDA results
    if 'eda_results' not in st.session_state or st.session_state.eda_results is None:
        st.error(" EDA results not found. Please complete Module 2 first.")
        if st.button(" Go to EDA"):
            st.session_state.current_step = 2
            st.rerun()
        return
    
    eda_results = st.session_state.eda_results
    
    # Detect issues
    if not issues_already_detected:
        with st.spinner("Analyzing data quality issues..."):
            issues = analyze_all_issues(df, target_column, eda_results)
            st.session_state.detected_issues = issues
    else:
        issues = st.session_state.detected_issues
    
    # Display summary
    total_issues = len(issues['all'])
    
    if total_issues == 0:
        st.success("""
            ##  No Issues Detected!
            
            Your dataset is in excellent condition:
            - No missing values
            - No significant outliers
            - No class imbalance
            - No data quality issues
        """)
        
        st.info("""
            ** Note:** Since no data quality issues were detected, we're skipping the preprocessing step 
            and will proceed directly to model training. Your data is already clean and ready to use!
        """)
        
        st.session_state.user_approvals = {}
        st.session_state.issues_completed = True
        st.session_state.skip_preprocessing = True  # Flag to skip Module 4
        
        # Navigation
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if st.button(" Back to EDA", use_container_width=True):
                st.session_state.current_step = 2
                st.rerun()
        with col3:
            if st.button("Continue to Model Training ", use_container_width=True, type="primary"):
                st.session_state.current_step = 5  # Skip to Module 5 (Training)
                st.rerun()
        return
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Issues", total_issues)
    with col2:
        st.metric("Critical", len(issues['critical']), delta=None if len(issues['critical']) == 0 else "Action Required", delta_color="inverse")
    with col3:
        st.metric("High Priority", len(issues['high']))
    with col4:
        st.metric("Medium/Low", len(issues['medium']) + len(issues['low']))
    
    st.markdown("---")
    
    # Display issues by severity
    user_selections = {}
    issue_counter = 1
    
    # Critical issues
    if len(issues['critical']) > 0:
        st.markdown("##  CRITICAL ISSUES (Must Address)")
        for issue in issues['critical']:
            selection = display_issue_card(issue, issue_counter, f"critical_{issue_counter}")
            if selection:
                user_selections[f"{issue['column']}_{issue['issue_type']}"] = selection
            issue_counter += 1
        st.markdown("---")
    
    # High priority
    if len(issues['high']) > 0:
        st.markdown("##  HIGH PRIORITY ISSUES")
        for issue in issues['high']:
            selection = display_issue_card(issue, issue_counter, f"high_{issue_counter}")
            if selection:
                user_selections[f"{issue['column']}_{issue['issue_type']}"] = selection
            issue_counter += 1
        st.markdown("---")
    
    # Medium/Low priority
    if len(issues['medium']) + len(issues['low']) > 0:
        st.markdown("##  MEDIUM & LOW PRIORITY ISSUES")
        for issue in issues['medium'] + issues['low']:
            selection = display_issue_card(issue, issue_counter, f"medlow_{issue_counter}")
            if selection:
                user_selections[f"{issue['column']}_{issue['issue_type']}"] = selection
            issue_counter += 1
        st.markdown("---")
    
    # Summary of selections
    st.markdown("##  Summary of Approved Fixes")
    
    if user_selections:
        # Group by action type
        drops = [sel['column'] for sel in user_selections.values() if sel.get('action') == 'drop']
        imputes = {sel['column']: sel['method'] for sel in user_selections.values() if sel.get('action') == 'impute'}
        outlier_fixes = {sel['column']: sel['action'] for sel in user_selections.values() if sel['issue_type'] == 'outliers'}
        
        if drops:
            st.info(f"**Drop Columns**: {', '.join(drops)}")
        if imputes:
            st.info(f"**Impute Missing**: {', '.join([f'{col} ({method})' for col, method in imputes.items()])}")
        if outlier_fixes:
            st.info(f"**Handle Outliers**: {', '.join([f'{col} ({action})' for col, action in outlier_fixes.items()])}")
        
        # Check for class imbalance fix
        for sel in user_selections.values():
            if sel.get('issue_type') == 'class_imbalance':
                techniques = []
                if sel.get('apply_smote'):
                    techniques.append("SMOTE")
                if sel.get('stratified_split'):
                    techniques.append("Stratified Split")
                if sel.get('class_weights'):
                    techniques.append("Class Weights")
                if techniques:
                    st.info(f"**Class Balancing**: {', '.join(techniques)}")
    else:
        st.warning("No fixes selected yet. Please review issues above.")
    
    # Store approvals
    st.session_state.user_approvals = user_selections
    st.session_state.issues_completed = True
    st.session_state.skip_preprocessing = False  # Don't skip if there are issues
    
    # Navigation
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button(" Back to EDA", use_container_width=True):
            st.session_state.current_step = 2
            st.rerun()
    with col3:
        if st.button("Continue to Preprocessing ", use_container_width=True, type="primary"):
            st.session_state.current_step = 4
            st.rerun()
