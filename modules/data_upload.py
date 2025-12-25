"""
Module 1: Dataset Upload & Basic Information
Handles CSV/Excel file upload and displays basic dataset information.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, Tuple
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import (
    check_file_size, 
    validate_dataframe,
    validate_classification_dataset,
    get_numeric_columns,
    get_categorical_columns
)


def upload_dataset() -> Optional[pd.DataFrame]:
    """
    Display file uploader and load dataset.
    
    Returns:
        Loaded DataFrame or None if upload fails
    """
    # Enhanced upload section with emojis and better copy
    st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(102,126,234,0.08) 0%, rgba(118,75,162,0.08) 100%); 
                    padding: 1.5rem; border-radius: 16px; margin-bottom: 2rem; border: 1px solid rgba(102,126,234,0.1);">
            <h2 style="margin-top: 0;"> Upload Your Dataset</h2>
            <p style="color: #6b7280; font-size: 1rem; margin-bottom: 0;">
                Upload a CSV or Excel file containing your <strong>classification dataset</strong>. 
                Your target variable should have discrete classes (not continuous values).
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['csv', 'xlsx', 'xls'],
        help="Maximum file size: 10MB. Supported formats: CSV, Excel (.xlsx, .xls)",
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        # Show file info
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"** Filename:** `{uploaded_file.name}`")
        with col2:
            file_size_mb = uploaded_file.size / (1024 * 1024)
            st.write(f"** Size:** `{file_size_mb:.2f} MB`")
        
        # Validate file size
        is_valid, message = check_file_size(uploaded_file, max_mb=10)
        
        if not is_valid:
            st.error(f" {message}")
            return None
        
        try:
            with st.spinner(' Loading your dataset...'):
                # Load the file based on extension
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:  # Excel file
                    df = pd.read_excel(uploaded_file)
            
            # Validate dataframe
            is_valid, message = validate_dataframe(df)
            
            if not is_valid:
                st.error(f" {message}")
                return None
            
            st.success(f" Dataset loaded successfully! Found {len(df):,} rows and {len(df.columns)} columns.")
            return df
            
        except Exception as e:
            st.error(f" Error loading file: {str(e)}")
            st.info(" **Troubleshooting Tips:**")
            st.markdown("""
                - Ensure your file is a valid CSV or Excel format
                - Check that the file is not corrupted
                - Verify column headers are properly formatted
                - Make sure there are no special characters in the file path
            """)
            return None
    
    return None


def display_basic_info(df: pd.DataFrame) -> None:
    """
    Display basic information about the dataset.
    
    Args:
        df: Pandas DataFrame
    """
    st.markdown("""
        <h2 style="margin-bottom: 1.5rem;"> Dataset Overview</h2>
    """, unsafe_allow_html=True)
    
    # Create three columns for metrics with gradient backgrounds
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Total Rows",
            value=f"{len(df):,}",
            help="Number of samples/observations in the dataset"
        )
    
    with col2:
        st.metric(
            label="Total Columns",
            value=len(df.columns),
            help="Number of features (including target variable)"
        )
    
    with col3:
        total_missing = df.isnull().sum().sum()
        missing_percentage = (total_missing / (len(df) * len(df.columns))) * 100
        st.metric(
            label="Missing Values",
            value=f"{total_missing:,}",
            delta=f"{missing_percentage:.1f}%",
            delta_color="inverse",
            help="Total number of missing values in the dataset"
        )
    
    # Display data preview with header
    st.markdown("###  Data Preview")
    st.caption("First 10 rows of your dataset")
    st.dataframe(df.head(10), use_container_width=True)


def display_column_info(df: pd.DataFrame) -> None:
    """
    Display detailed column information including data types.
    
    Args:
        df: Pandas DataFrame
    """
    st.subheader(" Column Information")
    
    # Create a summary dataframe
    column_info = pd.DataFrame({
        'Column Name': df.columns,
        'Data Type': df.dtypes.astype(str),
        'Non-Null Count': df.count(),
        'Null Count': df.isnull().sum(),
        'Null %': (df.isnull().sum() / len(df) * 100).round(2),
        'Unique Values': df.nunique()
    })
    
    # Add a category column
    def categorize_dtype(dtype):
        if dtype in ['int64', 'float64', 'int32', 'float32']:
            return ' Numerical'
        else:
            return ' Categorical'
    
    column_info['Category'] = column_info['Data Type'].apply(categorize_dtype)
    
    # Reorder columns
    column_info = column_info[[
        'Column Name', 'Category', 'Data Type', 
        'Non-Null Count', 'Null Count', 'Null %', 'Unique Values'
    ]]
    
    st.dataframe(column_info, use_container_width=True, hide_index=True)
    
    # Summary statistics
    col1, col2 = st.columns(2)
    
    with col1:
        numeric_cols = len(get_numeric_columns(df))
        st.info(f" **Numerical Columns**: {numeric_cols}")
    
    with col2:
        categorical_cols = len(get_categorical_columns(df))
        st.info(f" **Categorical Columns**: {categorical_cols}")


def display_summary_statistics(df: pd.DataFrame) -> None:
    """
    Display summary statistics for numerical features.
    
    Args:
        df: Pandas DataFrame
    """
    st.subheader(" Summary Statistics")
    
    numeric_cols = get_numeric_columns(df)
    
    if len(numeric_cols) == 0:
        st.warning(" No numerical columns found in the dataset.")
        return
    
    # Display statistics
    summary_stats = df[numeric_cols].describe().T
    summary_stats['missing'] = df[numeric_cols].isnull().sum()
    summary_stats['missing %'] = (df[numeric_cols].isnull().sum() / len(df) * 100).round(2)
    
    st.dataframe(summary_stats, use_container_width=True)
    
    st.write("**Legend:**")
    st.caption("""
    - **count**: Number of non-null values
    - **mean**: Average value
    - **std**: Standard deviation (measure of spread)
    - **min/max**: Minimum and maximum values
    - **25%, 50%, 75%**: Quartiles (percentiles)
    """)


def select_target_column(df: pd.DataFrame) -> Optional[str]:
    """
    Allow user to select the target column for classification.
    
    Args:
        df: Pandas DataFrame
        
    Returns:
        Name of selected target column
    """
    st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(79,172,254,0.08) 0%, rgba(0,242,254,0.08) 100%); 
                    padding: 1.5rem; border-radius: 16px; margin-bottom: 1.5rem; border: 1px solid rgba(79,172,254,0.1);">
            <h2 style="margin-top: 0;"> Select Target Column</h2>
            <p style="color: #6b7280; font-size: 1rem; margin-bottom: 0;">
                Choose the column you want to predict (your target variable). This should be a column 
                containing <strong>discrete classes or categories</strong>, not continuous numerical values.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Infer likely target column using helper from utils
    from utils.helpers import infer_target_column as infer_helper
    suggested_target = infer_helper(df)
    suggested_index = df.columns.tolist().index(suggested_target)
    
    target_column = st.selectbox(
        "Target Column",
        options=df.columns.tolist(),
        index=suggested_index,
        help=" Common target column names: 'target', 'label', 'class', 'category', 'outcome', etc.",
        label_visibility="collapsed"
    )
    
    if target_column:
        unique_count = df[target_column].nunique()
        st.markdown(f"""
            <div style="padding: 1rem; background: rgba(79,172,254,0.05); border-radius: 12px; border-left: 4px solid #4facfe;">
                <p style="margin: 0; font-weight: 500;">
                     Selected: <strong>{target_column}</strong> 
                    <span style="color: #6b7280;">({unique_count} unique values)</span>
                </p>
            </div>
        """, unsafe_allow_html=True)
        return target_column
    
    return None


def display_class_distribution(df: pd.DataFrame, target_column: str) -> None:
    """
    Display class distribution for the target variable.
    
    Args:
        df: Pandas DataFrame
        target_column: Name of the target column
    """
    st.subheader(" Class Distribution")
    
    if target_column not in df.columns:
        st.error(f" Column '{target_column}' not found in dataset")
        return
    
    # Calculate class distribution
    class_counts = df[target_column].value_counts().sort_index()
    class_percentages = (class_counts / len(df) * 100).round(2)
    
    # Create distribution table
    dist_df = pd.DataFrame({
        'Class': class_counts.index,
        'Count': class_counts.values,
        'Percentage': class_percentages.values
    })
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Number of Classes", len(class_counts))
    
    with col2:
        majority_class_pct = class_percentages.max()
        st.metric("Majority Class %", f"{majority_class_pct:.1f}%")
    
    with col3:
        minority_class_pct = class_percentages.min()
        st.metric("Minority Class %", f"{minority_class_pct:.1f}%")
    
    # Display distribution table
    st.write("#### Class Breakdown")
    st.dataframe(dist_df, use_container_width=True, hide_index=True)
    
    # Create visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Bar chart
        fig_bar = px.bar(
            dist_df,
            x='Class',
            y='Count',
            title='Class Distribution (Bar Chart)',
            labels={'Count': 'Number of Samples'},
            text='Count',
            color='Count',
            color_continuous_scale='blues'
        )
        fig_bar.update_traces(texttemplate='%{text}', textposition='outside')
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        # Pie chart
        fig_pie = px.pie(
            dist_df,
            values='Count',
            names='Class',
            title='Class Distribution (Pie Chart)',
            hole=0.3
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Check for class imbalance
    imbalance_ratio = majority_class_pct / minority_class_pct
    
    if imbalance_ratio > 2:
        st.warning(f"""
         **Class Imbalance Detected!**
        
        The majority class is {imbalance_ratio:.1f}x larger than the minority class.
        This may affect model performance. Consider using:
        - Class weighting
        - Resampling techniques (SMOTE, undersampling)
        - Stratified train/test split
        """)
    else:
        st.success(" Classes are relatively balanced.")


def run_module_1() -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Main function to run Module 1: Dataset Upload & Basic Info.
    
    Returns:
        Tuple of (DataFrame, target_column_name)
    """
    # Modern Hero Section
    st.markdown("""
        <div style="text-align: center; padding: 2rem 0 1rem 0;">
            <h1 style="font-size: 3.5rem; margin-bottom: 0.5rem;"> AutoML Classification System</h1>
            <p style="font-size: 1.25rem; color: #6b7280; font-weight: 400; max-width: 800px; margin: 0 auto;">
                Automate your entire ML pipeline from data upload to model comparison with our intelligent system
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Check if data is already loaded (for back navigation)
    data_already_loaded = (
        'df' in st.session_state and 
        st.session_state.df is not None and 
        'target_column' in st.session_state and 
        st.session_state.target_column is not None
    )
    
    if data_already_loaded:
        # Show option to use existing data or upload new
        st.info(" **Dataset already loaded!** You can continue with the current dataset or upload a new one.")
        
        col1, col2 = st.columns(2)
        with col1:
            use_existing = st.button(" View Current Dataset", use_container_width=True, type="primary")
        with col2:
            upload_new = st.button(" Upload New Dataset", use_container_width=True)
        
        if upload_new:
            # Clear existing data and show upload interface
            st.session_state.df = None
            st.session_state.target_column = None
            st.rerun()
        
        if use_existing or (not upload_new):
            # Display the already-loaded dataset
            df = st.session_state.df
            target_column = st.session_state.target_column
            
            st.markdown("---")
            # Fixed text colors - labels in gray, values in green
            st.markdown(f"""
                <div style="margin-bottom: 1rem;">
                    <span style="color: #9ca3af; font-size: 1.1rem;">Current Dataset: </span>
                    <span style="color: #22c55e; font-size: 1.1rem; font-weight: 600;">{len(df):,} rows × {len(df.columns)} columns</span>
                </div>
                <div style="margin-bottom: 1rem;">
                    <span style="color: #9ca3af; font-size: 1.1rem;">Target Column: </span>
                    <span style="color: #22c55e; font-size: 1.1rem; font-weight: 600;">{target_column}</span>
                </div>
            """, unsafe_allow_html=True)
            
            # Option to change target column without uploading new data
            with st.expander("Change Target Column", expanded=False):
                new_target = st.selectbox(
                    "Select new target column:",
                    options=df.columns.tolist(),
                    index=df.columns.tolist().index(target_column) if target_column in df.columns.tolist() else 0,
                    key="change_target_select"
                )
                if st.button("Apply New Target", key="apply_new_target"):
                    if new_target != target_column:
                        # Validate new target (function already imported at top of file)
                        is_valid, msg = validate_classification_dataset(df, new_target)
                        if is_valid:
                            st.session_state.target_column = new_target
                            # Reset downstream states since target changed
                            st.session_state.eda_completed = False
                            st.session_state.issues_completed = False
                            st.session_state.preprocessing_completed = False
                            st.session_state.training_completed = False
                            st.success(f"Target changed to '{new_target}'!")
                            st.rerun()
                        else:
                            st.error(msg)
            
            st.markdown("---")
            
            # Display all the analyses
            display_basic_info(df)
            st.markdown("---")
            
            display_column_info(df)
            st.markdown("---")
            
            display_summary_statistics(df)
            st.markdown("---")
            
            # Show target info
            st.markdown(f"""
                <div style="background: rgba(34, 197, 94, 0.08); 
                            padding: 1.5rem; border-radius: 16px; margin-bottom: 1.5rem; border: 1px solid rgba(34, 197, 94, 0.2);">
                    <h3 style="margin-top: 0; color: #22c55e;">Selected Target Column</h3>
                    <p style="color: #9ca3af; font-size: 1rem; margin-bottom: 0;">
                        Currently using <span style="color: #22c55e; font-weight: 600;">{target_column}</span> as the target variable.
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            # Display class distribution
            display_class_distribution(df, target_column)
            
            return df, target_column
    
    # Original upload flow (when no data is loaded)
    df = upload_dataset()
    
    if df is not None:
        # Select target column FIRST - before any analysis
        target_column = select_target_column(df)
        
        if target_column:
            # Validate classification dataset IMMEDIATELY - before showing ANY dataset info
            is_valid_classification, validation_message = validate_classification_dataset(df, target_column)
            
            if not is_valid_classification:
                # Show error and STOP - don't show any dataset information
                st.markdown("---")
                st.error(f"###  Invalid Dataset Type\n\n{validation_message}")
                st.info("""
                    ###  What to do?
                    
                    1. **Upload a different dataset** with a categorical target variable
                    2. **Select a different target column** that represents classes
                    3. Make sure your target has discrete categories (e.g., 'Yes/No', 'Cat A/B/C', 0/1, etc.)
                """)
                return None, None
            
            # If valid classification, show success and continue with analysis
            st.markdown("---")
            st.success(validation_message)
            st.markdown("---")
            
            # NOW show dataset analysis
            display_basic_info(df)
            st.markdown("---")
            
            display_column_info(df)
            st.markdown("---")
            
            display_summary_statistics(df)
            st.markdown("---")
            
            # Display class distribution
            display_class_distribution(df, target_column)
            
            return df, target_column
    
    return None, None
