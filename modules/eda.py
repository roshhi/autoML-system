"""
Module 2: Automated Exploratory Data Analysis (EDA)

This module provides comprehensive EDA functionality including:
- Missing value analysis and visualization
- Outlier detection (IQR and Z-score methods)
- Correlation matrix generation and visualization
- Distribution analysis for numerical and categorical features
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict, List, Tuple, Optional
import streamlit as st


def analyze_missing_values(df: pd.DataFrame) -> Dict:
    """
    Analyze missing values in the dataset.
    
    Args:
        df: Pandas DataFrame
        
    Returns:
        Dictionary containing:
        - missing_counts: Series of missing counts per column
        - missing_percentages: Series of missing % per column
        - total_missing: Total missing values
        - global_percentage: Overall missing %
        - columns_with_missing: List of columns with missing data
    """
    # Calculate missing values
    missing_counts = df.isnull().sum()
    total_cells = df.shape[0] * df.shape[1]
    total_missing = missing_counts.sum()
    
    # Calculate percentages
    missing_percentages = (missing_counts / len(df)) * 100
    global_percentage = (total_missing / total_cells) * 100
    
    # Get columns with missing values
    columns_with_missing = missing_counts[missing_counts > 0].index.tolist()
    
    return {
        'missing_counts': missing_counts,
        'missing_percentages': missing_percentages,
        'total_missing': int(total_missing),
        'global_percentage': global_percentage,
        'columns_with_missing': columns_with_missing,
        'num_affected_columns': len(columns_with_missing)
    }


def plot_missing_values(df: pd.DataFrame, missing_data: Dict) -> Tuple[go.Figure, go.Figure]:
    """
    Create visualizations for missing values.
    
    Args:
        df: DataFrame
        missing_data: Output from analyze_missing_values()
        
    Returns:
        Tuple of (bar_chart_fig, heatmap_fig)
    """
    # Bar chart of missing percentages (only columns with missing data)
    columns_with_missing = missing_data['columns_with_missing']
    
    if len(columns_with_missing) > 0:
        missing_pct_filtered = missing_data['missing_percentages'][columns_with_missing].sort_values(ascending=True)
        
        bar_fig = go.Figure()
        bar_fig.add_trace(go.Bar(
            y=missing_pct_filtered.index,
            x=missing_pct_filtered.values,
            orientation='h',
            marker=dict(
                color=missing_pct_filtered.values,
                colorscale='YlOrRd',
                showscale=True,
                colorbar=dict(title="% Missing")
            ),
            text=[f"{val:.1f}%" for val in missing_pct_filtered.values],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Missing: %{x:.2f}%<extra></extra>'
        ))
        
        bar_fig.update_layout(
            title="Missing Value Percentage by Column",
            xaxis_title="Percentage Missing (%)",
            yaxis_title="Column",
            height=max(400, len(columns_with_missing) * 30),
            showlegend=False,
            template="plotly_white"
        )
    else:
        # No missing values - create empty figure with message
        bar_fig = go.Figure()
        bar_fig.add_annotation(
            text=" No missing values detected in any column!",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20, color="green")
        )
        bar_fig.update_layout(height=300)
    
    # Heatmap showing missing value patterns
    # Create binary matrix (1 = missing, 0 = present)
    missing_matrix = df.isnull().astype(int)
    
    if len(columns_with_missing) > 0:
        # Only show columns with missing values
        missing_matrix_filtered = missing_matrix[columns_with_missing]
        
        heatmap_fig = go.Figure(data=go.Heatmap(
            z=missing_matrix_filtered.T.values,
            x=missing_matrix_filtered.index,
            y=missing_matrix_filtered.columns,
            colorscale=[[0, '#2ecc71'], [1, '#e74c3c']],
            showscale=True,
            colorbar=dict(
                title="Missing",
                tickvals=[0, 1],
                ticktext=['Present', 'Missing']
            ),
            hovertemplate='Row: %{x}<br>Column: %{y}<br>Status: %{z}<extra></extra>'
        ))
        
        heatmap_fig.update_layout(
            title="Missing Value Pattern Matrix",
            xaxis_title="Row Index",
            yaxis_title="Column",
            height=max(400, len(columns_with_missing) * 40),
            template="plotly_white"
        )
    else:
        # No missing values
        heatmap_fig = go.Figure()
        heatmap_fig.add_annotation(
            text=" Complete dataset - no missing values!",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20, color="green")
        )
        heatmap_fig.update_layout(height=300)
    
    return bar_fig, heatmap_fig


def detect_outliers_iqr(df: pd.DataFrame, columns: List[str]) -> Dict:
    """
    Detect outliers using IQR (Interquartile Range) method.
    
    Args:
        df: DataFrame
        columns: List of numerical columns to check
        
    Returns:
        Dictionary with outlier information per column
    """
    outlier_results = {}
    
    for col in columns:
        data = df[col].dropna()
        
        if len(data) == 0:
            continue
        
        # Calculate quartiles
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        # Calculate bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Find outliers
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        outlier_count = len(outliers)
        outlier_percentage = (outlier_count / len(data)) * 100
        
        outlier_results[col] = {
            'method': 'IQR',
            'outlier_count': outlier_count,
            'outlier_percentage': outlier_percentage,
            'outlier_indices': outliers.index.tolist(),
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR
        }
    
    return outlier_results


def detect_outliers_zscore(df: pd.DataFrame, columns: List[str], threshold: float = 3.0) -> Dict:
    """
    Detect outliers using Z-score method.
    
    Args:
        df: DataFrame
        columns: Numerical columns to check
        threshold: Z-score threshold (default: 3.0)
        
    Returns:
        Dictionary with outlier information per column
    """
    outlier_results = {}
    
    for col in columns:
        data = df[col].dropna()
        
        if len(data) == 0:
            continue
        
        # Calculate z-scores
        z_scores = np.abs(stats.zscore(data))
        
        # Find outliers
        outlier_mask = z_scores > threshold
        outliers = data[outlier_mask]
        outlier_count = len(outliers)
        outlier_percentage = (outlier_count / len(data)) * 100
        
        outlier_results[col] = {
            'method': 'Z-Score',
            'outlier_count': outlier_count,
            'outlier_percentage': outlier_percentage,
            'outlier_indices': outliers.index.tolist(),
            'threshold': threshold,
            'mean': data.mean(),
            'std': data.std()
        }
    
    return outlier_results


def plot_outliers_boxplot(df: pd.DataFrame, columns: List[str]) -> go.Figure:
    """
    Create box plots showing outliers for numerical columns.
    
    Args:
        df: DataFrame
        columns: Columns to plot
        
    Returns:
        Plotly box plot figure
    """
    # Determine grid layout
    n_cols = min(3, len(columns))
    n_rows = (len(columns) + n_cols - 1) // n_cols
    
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=columns,
        vertical_spacing=0.1,
        horizontal_spacing=0.08
    )
    
    for idx, col in enumerate(columns):
        row = (idx // n_cols) + 1
        col_pos = (idx % n_cols) + 1
        
        fig.add_trace(
            go.Box(
                y=df[col],
                name=col,
                marker=dict(color='rgba(102, 126, 234, 0.6)'),
                boxmean='sd',  # Show mean and std deviation
                hovertemplate='<b>%{y}</b><extra></extra>'
            ),
            row=row,
            col=col_pos
        )
    
    fig.update_layout(
        title_text="Box Plots - Outlier Detection",
        showlegend=False,
        height=300 * n_rows,
        template="plotly_white"
    )
    
    return fig


def generate_correlation_matrix(df: pd.DataFrame, numerical_columns: List[str]) -> Tuple[pd.DataFrame, go.Figure, List[Tuple]]:
    """
    Calculate and visualize correlation matrix.
    
    Args:
        df: DataFrame
        numerical_columns: List of numerical columns
        
    Returns:
        Tuple of (correlation_df, heatmap_fig, high_corr_pairs)
    """
    # Calculate correlation matrix
    corr_matrix = df[numerical_columns].corr()
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu_r',
        zmid=0,
        text=corr_matrix.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation"),
        hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Correlation Matrix Heatmap",
        xaxis_title="Features",
        yaxis_title="Features",
        height=max(500, len(numerical_columns) * 40),
        width=max(600, len(numerical_columns) * 40),
        template="plotly_white"
    )
    
    # Find highly correlated pairs (> 0.8, excluding diagonal)
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_value = corr_matrix.iloc[i, j]
            if abs(corr_value) > 0.8:
                high_corr_pairs.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_value
                ))
    
    return corr_matrix, fig, high_corr_pairs


def calculate_distribution_stats(df: pd.DataFrame, column: str) -> Dict:
    """
    Calculate distribution statistics for a numerical column.
    
    Args:
        df: DataFrame
        column: Column name
        
    Returns:
        Dictionary with statistics
    """
    data = df[column].dropna()
    
    return {
        'mean': data.mean(),
        'median': data.median(),
        'std': data.std(),
        'min': data.min(),
        'max': data.max(),
        'skewness': data.skew(),
        'kurtosis': data.kurtosis(),
        'q25': data.quantile(0.25),
        'q75': data.quantile(0.75)
    }


def plot_numerical_distributions(df: pd.DataFrame, columns: List[str]) -> go.Figure:
    """
    Create distribution plots (histograms) for numerical features.
    
    Args:
        df: DataFrame
        columns: Numerical columns to plot
        
    Returns:
        Plotly figure with histogram subplots
    """
    # Determine grid layout
    n_cols = min(2, len(columns))
    n_rows = (len(columns) + n_cols - 1) // n_cols
    
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[f"{col}" for col in columns],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    for idx, col in enumerate(columns):
        row = (idx // n_cols) + 1
        col_pos = (idx % n_cols) + 1
        
        # Get statistics
        stats_dict = calculate_distribution_stats(df, col)
        
        # Add histogram
        fig.add_trace(
            go.Histogram(
                x=df[col],
                name=col,
                marker=dict(
                    color='rgba(102, 126, 234, 0.7)',
                    line=dict(color='rgba(102, 126, 234, 1)', width=1)
                ),
                hovertemplate='Value: %{x}<br>Count: %{y}<extra></extra>',
                showlegend=False
            ),
            row=row,
            col=col_pos
        )
        
        # Add mean line
        fig.add_vline(
            x=stats_dict['mean'],
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {stats_dict['mean']:.2f}",
            annotation_position="top",
            row=row,
            col=col_pos
        )
    
    fig.update_layout(
        title_text="Distribution of Numerical Features",
        showlegend=False,
        height=400 * n_rows,
        template="plotly_white"
    )
    
    return fig


def plot_categorical_distributions(df: pd.DataFrame, columns: List[str], max_categories: int = 20) -> List[go.Figure]:
    """
    Create bar charts for categorical features.
    
    Args:
        df: DataFrame
        columns: Categorical columns to plot
        max_categories: Maximum categories to display per plot
        
    Returns:
        List of Plotly bar chart figures
    """
    figures = []
    
    for col in columns:
        value_counts = df[col].value_counts().head(max_categories)
        
        fig = go.Figure(data=[
            go.Bar(
                x=value_counts.index.astype(str),
                y=value_counts.values,
                marker=dict(
                    color=value_counts.values,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Count")
                ),
                text=value_counts.values,
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title=f"Distribution of '{col}' (Top {len(value_counts)} categories)",
            xaxis_title=col,
            yaxis_title="Count",
            height=400,
            template="plotly_white",
            showlegend=False
        )
        
        figures.append(fig)
    
    return figures


def run_module_2(df: pd.DataFrame, target_column: str) -> None:
    """
    Main function to run Module 2: Automated EDA.
    Displays all EDA results in Streamlit.
    
    Args:
        df: DataFrame
        target_column: Name of target column
    """
    st.markdown("""
        <div style="text-align: center; padding: 1.5rem 0;">
            <h1 style="font-size: 3rem; margin-bottom: 0.5rem;"> Exploratory Data Analysis</h1>
            <p style="font-size: 1.15rem; color: #6b7280; font-weight: 400;">
                Automated comprehensive analysis of your dataset
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Check if EDA has already been run
    eda_already_run = (
        'eda_results' in st.session_state and 
        st.session_state.eda_results is not None and
        'eda_completed' in st.session_state and
        st.session_state.eda_completed == True
    )
    
    if eda_already_run:
        st.info(" **EDA already completed!** Displaying cached results. Click 'Re-run EDA' to analyze again.")
        
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button(" Re-run EDA", use_container_width=True):
                # Clear EDA results
                st.session_state.eda_completed = False
                st.session_state.eda_results = None
                st.rerun()
    
    # Get numerical and categorical columns (excluding target)
    from utils.helpers import get_numeric_columns, get_categorical_columns
    
    all_numerical = get_numeric_columns(df)
    all_categorical = get_categorical_columns(df)
    
    # Exclude target from analyses where appropriate
    numerical_features = [col for col in all_numerical if col != target_column]
    categorical_features = [col for col in all_categorical if col != target_column]
    
    # === 1. MISSING VALUE ANALYSIS ===
    st.markdown("##  Missing Value Analysis")
    
    with st.spinner("Analyzing missing values..."):
        missing_data = analyze_missing_values(df)
        
        # Display summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Total Missing Values",
                f"{missing_data['total_missing']:,}",
                help="Total number of missing cells in the dataset"
            )
        with col2:
            st.metric(
                "Global Missing %",
                f"{missing_data['global_percentage']:.2f}%",
                help="Percentage of all cells that are missing"
            )
        with col3:
            st.metric(
                "Affected Columns",
                missing_data['num_affected_columns'],
                help="Number of columns with at least one missing value"
            )
        
        # Visualizations
        if missing_data['num_affected_columns'] > 0:
            bar_fig, heatmap_fig = plot_missing_values(df, missing_data)
            
            st.plotly_chart(bar_fig, use_container_width=True)
            st.plotly_chart(heatmap_fig, use_container_width=True)
            
            # Missing value table
            st.markdown("### Missing Value Details")
            missing_df = pd.DataFrame({
                'Column': missing_data['columns_with_missing'],
                'Missing Count': [missing_data['missing_counts'][col] for col in missing_data['columns_with_missing']],
                'Missing %': [f"{missing_data['missing_percentages'][col]:.2f}%" for col in missing_data['columns_with_missing']]
            })
            st.dataframe(missing_df, use_container_width=True)
        else:
            st.success(" **Excellent!** No missing values detected in the dataset.")
    
    st.markdown("---")
    
    # === 2. OUTLIER DETECTION ===
    if len(numerical_features) > 0:
        st.markdown("##  Outlier Detection")
        
        # Method selection
        outlier_method = st.radio(
            "Select outlier detection method:",
            options=["IQR (Interquartile Range)", "Z-Score", "Both"],
            horizontal=True,
            help="IQR: Values beyond Q1-1.5*IQR or Q3+1.5*IQR | Z-Score: |z-score| > 3"
        )
        
        with st.spinner("Detecting outliers..."):
            # Detect outliers based on method
            if outlier_method in ["IQR (Interquartile Range)", "Both"]:
                iqr_results = detect_outliers_iqr(df, numerical_features)
                
                # Summary
                total_outliers_iqr = sum([res['outlier_count'] for res in iqr_results.values()])
                st.info(f"**IQR Method**: Detected {total_outliers_iqr:,} outliers across {len([r for r in iqr_results.values() if r['outlier_count'] > 0])} columns")
                
                # Outlier table
                if total_outliers_iqr > 0:
                    iqr_df = pd.DataFrame({
                        'Column': list(iqr_results.keys()),
                        'Outliers': [res['outlier_count'] for res in iqr_results.values()],
                        'Outlier %': [f"{res['outlier_percentage']:.2f}%" for res in iqr_results.values()],
                        'Lower Bound': [f"{res['lower_bound']:.2f}" for res in iqr_results.values()],
                        'Upper Bound': [f"{res['upper_bound']:.2f}" for res in iqr_results.values()]
                    })
                    st.dataframe(iqr_df[iqr_df['Outliers'] > 0], use_container_width=True)
            
            if outlier_method in ["Z-Score", "Both"]:
                zscore_results = detect_outliers_zscore(df, numerical_features, threshold=3.0)
                
                # Summary
                total_outliers_z = sum([res['outlier_count'] for res in zscore_results.values()])
                st.info(f"**Z-Score Method**: Detected {total_outliers_z:,} outliers across {len([r for r in zscore_results.values() if r['outlier_count'] > 0])} columns")
                
                # Outlier table
                if total_outliers_z > 0:
                    z_df = pd.DataFrame({
                        'Column': list(zscore_results.keys()),
                        'Outliers': [res['outlier_count'] for res in zscore_results.values()],
                        'Outlier %': [f"{res['outlier_percentage']:.2f}%" for res in zscore_results.values()],
                        'Mean': [f"{res['mean']:.2f}" for res in zscore_results.values()],
                        'Std Dev': [f"{res['std']:.2f}" for res in zscore_results.values()]
                    })
                    st.dataframe(z_df[z_df['Outliers'] > 0], use_container_width=True)
            
            # Box plots
            st.markdown("### Box Plots Visualization")
            boxplot_fig = plot_outliers_boxplot(df, numerical_features)
            st.plotly_chart(boxplot_fig, use_container_width=True)
        
        st.markdown("---")
    
    # === 3. CORRELATION ANALYSIS ===
    if len(numerical_features) >= 2:
        st.markdown("##  Correlation Analysis")
        
        with st.spinner("Calculating correlations..."):
            corr_matrix, corr_fig, high_corr_pairs = generate_correlation_matrix(df, numerical_features)
            
            # Display heatmap
            st.plotly_chart(corr_fig, use_container_width=True)
            
            # High correlation pairs
            if len(high_corr_pairs) > 0:
                st.warning(f" **Found {len(high_corr_pairs)} highly correlated pairs (|r| > 0.8)**")
                high_corr_df = pd.DataFrame(high_corr_pairs, columns=['Feature 1', 'Feature 2', 'Correlation'])
                high_corr_df['Correlation'] = high_corr_df['Correlation'].apply(lambda x: f"{x:.3f}")
                st.dataframe(high_corr_df, use_container_width=True)
                
                st.info("""
                     **Multicollinearity Warning**
                    
                    High correlations between features may cause issues in some models. Consider:
                    - Removing one of the correlated features
                    - Using dimensionality reduction (PCA)
                    - Using regularization techniques
                """)
            else:
                st.success(" No highly correlated feature pairs detected.")
        
        st.markdown("---")
    
    # === 4. DISTRIBUTION ANALYSIS - NUMERICAL ===
    if len(numerical_features) > 0:
        st.markdown("##  Numerical Feature Distributions")
        
        with st.spinner("Generating distribution plots..."):
            dist_fig = plot_numerical_distributions(df, numerical_features)
            st.plotly_chart(dist_fig, use_container_width=True)
            
            # Distribution statistics table
            st.markdown("### Distribution Statistics")
            stats_data = []
            for col in numerical_features:
                stats = calculate_distribution_stats(df, col)
                stats_data.append({
                    'Feature': col,
                    'Mean': f"{stats['mean']:.2f}",
                    'Median': f"{stats['median']:.2f}",
                    'Std Dev': f"{stats['std']:.2f}",
                    'Skewness': f"{stats['skewness']:.2f}",
                    'Kurtosis': f"{stats['kurtosis']:.2f}"
                })
            
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True)
            
            # Skewness interpretation
            st.markdown("""
                ** Skewness Interpretation:**
                - **< -1 or > 1**: Highly skewed
                - **-1 to -0.5 or 0.5 to 1**: Moderately skewed
                - **-0.5 to 0.5**: Approximately symmetric
            """)
        
        st.markdown("---")
    
    # === 5. DISTRIBUTION ANALYSIS - CATEGORICAL ===
    if len(categorical_features) > 0:
        st.markdown("##  Categorical Feature Distributions")
        
        with st.spinner("Analyzing categorical features..."):
            cat_figs = plot_categorical_distributions(df, categorical_features)
            
            for fig in cat_figs:
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
    
    # === SUMMARY & NEXT STEPS ===
    st.markdown("##  EDA Complete!")
    
    st.success("""
        **Analysis Summary:**
        -  Missing value analysis completed
        -  Outlier detection performed
        -  Correlation analysis done
        -  Distribution analysis finished
        
        **Next Steps:**
        - Review the findings above
        - Proceed to Issue Detection & User Approval (Module 3)
        - Make informed preprocessing decisions
    """)
    
    # Store results in session state for later use
    st.session_state.eda_results = {
        'missing_data': missing_data,
        'numerical_features': numerical_features,
        'categorical_features': categorical_features,
        'outlier_results_iqr': iqr_results if len(numerical_features) > 0 and outlier_method in ["IQR (Interquartile Range)", "Both"] else {},
        'correlation_matrix': corr_matrix if len(numerical_features) >= 2 else None,
        'high_corr_pairs': high_corr_pairs if len(numerical_features) >= 2 else []
    }
    
    # Navigation buttons
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button(" Back to Dataset Info", use_container_width=True):
            st.session_state.current_step = 1
            st.rerun()
    with col3:
        if st.button("Continue to Issues ", use_container_width=True):
            st.session_state.current_step = 3
            st.rerun()

