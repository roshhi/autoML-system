"""
Module 6: Model Comparison Dashboard

This module provides comprehensive visualizations and analysis to compare
all trained models and help users select the best model.
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Tuple, Any
import io


def create_metric_comparison_chart(results: Dict, metric_key: str, metric_name: str) -> go.Figure:
    """
    Create bar chart comparing models on a single metric.
    
    Args:
        results: Training results from Module 5
        metric_key: Key in metrics dict ('accuracy', 'precision', etc.)
        metric_name: Display name for the metric
        
    Returns:
        Plotly bar chart figure
    """
    models = []
    values = []
    
    for result in results.values():
        models.append(result['display_name'])
        values.append(result['metrics'][metric_key])
    
    # Sort by value
    sorted_pairs = sorted(zip(models, values), key=lambda x: x[1], reverse=True)
    models, values = zip(*sorted_pairs)
    
    # Color scale: green for high, red for low
    colors = px.colors.sample_colorscale("RdYlGn", [v for v in values])
    
    fig = go.Figure(data=[
        go.Bar(
            x=models,
            y=values,
            marker_color=colors,
            text=[f'{v:.3f}' for v in values],
            textposition='outside',
            hovertemplate='%{x}<br>%{y:.4f}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title=f'{metric_name} Comparison',
        xaxis_title='Model',
        yaxis_title=metric_name,
        yaxis_range=[0, max(values) * 1.1],
        template='plotly_white',
        height=400,
        showlegend=False
    )
    
    return fig


def create_radar_chart(results: Dict) -> go.Figure:
    """
    Create radar/spider chart showing all metrics for all models.
    
    Returns:
        Plotly radar chart figure
    """
    categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set2
    
    for idx, (model_key, result) in enumerate(results.items()):
        metrics = result['metrics']
        values = [
            metrics['accuracy'],
            metrics['precision'],
            metrics['recall'],
            metrics['f1_score']
        ]
        
        # Close the polygon
        values_closed = values + [values[0]]
        categories_closed = categories + [categories[0]]
        
        fig.add_trace(go.Scatterpolar(
            r=values_closed,
            theta=categories_closed,
            fill='toself',
            name=result['display_name'],
            line_color=colors[idx % len(colors)]
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        title='All Metrics Radar Chart',
        template='plotly_white',
        height=500
    )
    
    return fig


def create_training_time_chart(results: Dict) -> go.Figure:
    """
    Create bar chart showing training time comparison.
    
    Returns:
        Plotly bar chart figure
    """
    models = []
    times = []
    
    for result in results.values():
        models.append(result['display_name'])
        times.append(result['training_time'])
    
    # Sort by time (ascending - fastest first)
    sorted_pairs = sorted(zip(models, times), key=lambda x: x[1])
    models, times = zip(*sorted_pairs)
    
    # Color: green for fast, red for slow
    max_time = max(times)
    colors = ['rgb(0, 200, 0)' if t < max_time * 0.3 else 
              'rgb(255, 165, 0)' if t < max_time * 0.7 else 
              'rgb(255, 0, 0)' for t in times]
    
    fig = go.Figure(data=[
        go.Bar(
            y=models,
            x=times,
            orientation='h',
            marker_color=colors,
            text=[f'{t:.2f}s' for t in times],
            textposition='outside',
            hovertemplate='%{y}<br>%{x:.3f}s<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title='Training Time Comparison',
        xaxis_title='Training Time (seconds)',
        yaxis_title='Model',
        template='plotly_white',
        height=400,
        showlegend=False
    )
    
    return fig


def create_roc_curves(results: Dict, X_test: pd.DataFrame, y_test: pd.Series) -> go.Figure:
    """
    Create ROC curves for all models (binary classification only).
    
    Returns:
        Plotly line chart with ROC curves
    """
    from sklearn.metrics import roc_curve, auc
    
    fig = go.Figure()
    
    # Add diagonal (random classifier)
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        line=dict(dash='dash', color='gray'),
        name='Random (AUC = 0.5)',
        showlegend=True
    ))
    
    colors = px.colors.qualitative.Set1
    
    for idx, (model_key, result) in enumerate(results.items()):
        model = result['model']
        
        # Get probabilities or decision function
        try:
            if hasattr(model, 'predict_proba'):
                y_score = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, 'decision_function'):
                y_score = model.decision_function(X_test)
            else:
                continue
            
            fpr, tpr, _ = roc_curve(y_test, y_score)
            roc_auc = auc(fpr, tpr)
            
            fig.add_trace(go.Scatter(
                x=fpr,
                y=tpr,
                mode='lines',
                name=f'{result["display_name"]} (AUC = {roc_auc:.3f})',
                line=dict(color=colors[idx % len(colors)])
            ))
        except:
            continue
    
    fig.update_layout(
        title='ROC Curves Comparison',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        template='plotly_white',
        height=500,
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1])
    )
    
    return fig


def get_feature_importance(model: Any, feature_names: List[str], top_n: int = 10) -> pd.DataFrame:
    """
    Extract feature importance from tree-based models.
    
    Returns:
        DataFrame with feature names and importance scores
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        
        # Sort and get top N
        importance_df = importance_df.sort_values('Importance', ascending=False).head(top_n)
        
        return importance_df
    
    return None


def create_feature_importance_chart(importance_df: pd.DataFrame, model_name: str) -> go.Figure:
    """
    Create horizontal bar chart for feature importance.
    
    Returns:
        Plotly bar chart
    """
    fig = go.Figure(data=[
        go.Bar(
            y=importance_df['Feature'][::-1],
            x=importance_df['Importance'][::-1],
            orientation='h',
            marker_color='skyblue',
            text=[f'{v:.3f}' for v in importance_df['Importance'][::-1]],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title=f'{model_name} - Top {len(importance_df)} Most Important Features',
        xaxis_title='Importance Score',
        yaxis_title='Feature',
        template='plotly_white',
        height=400,
        showlegend=False
    )
    
    return fig


def create_rankings_table(results: Dict) -> pd.DataFrame:
    """
    Create comprehensive rankings table.
    
    Returns:
        DataFrame with rankings by different criteria
    """
    rankings = []
    
    for model_key, result in results.items():
        metrics = result['metrics']
        rankings.append({
            'Model': result['display_name'],
            'F1-Score': metrics['f1_score'],
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'ROC-AUC': metrics['roc_auc'] if metrics['roc_auc'] else 0,
            'Training Time': result['training_time']
        })
    
    df = pd.DataFrame(rankings)
    
    # Add ranks
    df['F1 Rank'] = df['F1-Score'].rank(ascending=False).astype(int)
    df['Accuracy Rank'] = df['Accuracy'].rank(ascending=False).astype(int)
    df['Speed Rank'] = df['Training Time'].rank(ascending=True).astype(int)  # Lower is better
    
    return df


def generate_csv_export(results: Dict) -> str:
    """
    Generate CSV string with all results.
    
    Returns:
        CSV string ready for download
    """
    export_data = []
    
    for model_key, result in results.items():
        metrics = result['metrics']
        row = {
            'Model': result['display_name'],
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1_score'],
            'ROC-AUC': metrics['roc_auc'] if metrics['roc_auc'] else 'N/A',
            'Training Time (s)': result['training_time'],
            'Best Parameters': str(result['best_params']) if result['best_params'] else 'None'
        }
        export_data.append(row)
    
    df = pd.DataFrame(export_data)
    
    # Convert to CSV
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue()


# ==================== UI FUNCTION ====================

def run_module_6() -> None:
    """
    Main function for Module 6: Model Comparison Dashboard.
    """
    st.markdown("""
        <div style="text-align: center; padding: 1.5rem 0;">
            <h1 style="font-size: 3rem; margin-bottom: 0.5rem;"> Model Comparison Dashboard</h1>
            <p style="font-size: 1.15rem; color: #6b7280; font-weight: 400;">
                Compare models and select the best performer
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Check if training is complete
    if not st.session_state.get('training_completed', False):
        st.error(" No trained models found. Please complete model training first.")
        if st.button(" Go to Training"):
            st.session_state.current_step = 5
            st.rerun()
        return
    
    results = st.session_state.trained_models
    
    # Get preprocessed data for ROC curves
    preprocessed_data = st.session_state.get('preprocessed_data')
    skip_preprocessing = st.session_state.get('skip_preprocessing', False)
    
    # ===== RANKINGS TABLE =====
    st.markdown("##  Overall Rankings")
    
    rankings_df = create_rankings_table(results)
    
    # Sort using proper multi-criteria: F1 (desc) -> Accuracy (desc) -> Time (asc)
    rankings_df_sorted = rankings_df.sort_values(
        by=['F1-Score', 'Accuracy', 'Training Time'],
        ascending=[False, False, True]  # F1 high, Acc high, Time low
    )
    
    # Select columns for display
    rankings_display = rankings_df_sorted[['Model', 'F1-Score', 'Accuracy', 'Training Time', 'F1 Rank', 'Accuracy Rank', 'Speed Rank']].copy()
    
    # Format for display
    rankings_display['F1-Score'] = rankings_display['F1-Score'].apply(lambda x: f'{x:.3f}')
    rankings_display['Accuracy'] = rankings_display['Accuracy'].apply(lambda x: f'{x*100:.2f}%')
    rankings_display['Training Time'] = rankings_display['Training Time'].apply(lambda x: f'{x:.2f}s')
    
    st.dataframe(rankings_display, use_container_width=True, hide_index=True)
    
    # Best model highlight (first row after sorting)
    best_model = rankings_df_sorted.iloc[0]
    st.success(f" **Best Overall Model**: {best_model['Model']} (Ranked by F1-Score  Accuracy  Speed)")

    
    st.markdown("---")
    
    # ===== CSV DOWNLOAD =====
    st.markdown("##  Download Results")
    
    csv_data = generate_csv_export(results)
    
    st.download_button(
        label=" Download Full Results (CSV)",
        data=csv_data,
        file_name="model_comparison_results.csv",
        mime="text/csv",
        use_container_width=True
    )
    
    st.markdown("---")
    
    # ===== VISUALIZATIONS =====
    st.markdown("##  Performance Visualizations")
    
    # Tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs([" Radar Chart", " Metric Charts", " ROC Curves", " Training Time"])
    
    with tab1:
        st.markdown("### All Metrics Comparison")
        st.caption("Easily compare all models across all metrics at once")
        radar_fig = create_radar_chart(results)
        st.plotly_chart(radar_fig, use_container_width=True)
    
    with tab2:
        st.markdown("### Individual Metric Comparisons")
        
        col1, col2 = st.columns(2)
        
        with col1:
            accuracy_fig = create_metric_comparison_chart(results, 'accuracy', 'Accuracy')
            st.plotly_chart(accuracy_fig, use_container_width=True)
            
            recall_fig = create_metric_comparison_chart(results, 'recall', 'Recall')
            st.plotly_chart(recall_fig, use_container_width=True)
        
        with col2:
            precision_fig = create_metric_comparison_chart(results, 'precision', 'Precision')
            st.plotly_chart(precision_fig, use_container_width=True)
            
            f1_fig = create_metric_comparison_chart(results, 'f1_score', 'F1-Score')
            st.plotly_chart(f1_fig, use_container_width=True)
    
    with tab3:
        # Check if binary classification
        if not skip_preprocessing and preprocessed_data:
            y_test = preprocessed_data['y_test']
            X_test = preprocessed_data['X_test']
            n_classes = len(np.unique(y_test))
        else:
            n_classes = 3  # Default to multi-class to skip
        
        if n_classes == 2:
            st.markdown("### ROC Curves (Binary Classification)")
            st.caption("Compare model discrimination ability - higher AUC is better")
            
            if not skip_preprocessing and preprocessed_data:
                roc_fig = create_roc_curves(results, X_test, y_test)
                st.plotly_chart(roc_fig, use_container_width=True)
            else:
                st.info("ROC curves require preprocessed data")
        else:
            st.info(" ROC curves are only available for binary classification problems. Your dataset has multiple classes.")
    
    with tab4:
        st.markdown("### Training Time Comparison")
        st.caption("Faster models are better for production and iteration")
        time_fig = create_training_time_chart(results)
        st.plotly_chart(time_fig, use_container_width=True)
        
        # Show fastest model
        fastest_model = min(results.values(), key=lambda x: x['training_time'])
        st.success(f" **Fastest Model**: {fastest_model['display_name']} ({fastest_model['training_time']:.2f}s)")
    
    st.markdown("---")
    
    # ===== FEATURE IMPORTANCE =====
    st.markdown("##  Feature Importance Analysis")
    st.caption("Available for tree-based models (Decision Tree, Random Forest)")
    
    tree_models = ['decision_tree', 'random_forest']
    feature_names = preprocessed_data.get('feature_names') if preprocessed_data else None
    
    if feature_names:
        importance_shown = False
        
        for model_key in tree_models:
            if model_key in results:
                result = results[model_key]
                importance_df = get_feature_importance(result['model'], feature_names, top_n=10)
                
                if importance_df is not None:
                    importance_shown = True
                    with st.expander(f"**{result['display_name']}** - Feature Importance"):
                        imp_fig = create_feature_importance_chart(importance_df, result['display_name'])
                        st.plotly_chart(imp_fig, use_container_width=True)
                        
                        st.dataframe(importance_df, use_container_width=True, hide_index=True)
        
        if not importance_shown:
            st.info("No tree-based models were trained or selected")
    else:
        st.info("Feature importance requires preprocessed data")
    
    st.markdown("---")
    
    # ===== RECOMMENDATIONS =====
    st.markdown("##  Recommendations")
    
    # Sort results by our multi-criteria ranking to get true best
    results_sorted = sorted(
        results.items(),
        key=lambda x: (
            -x[1]['metrics']['f1_score'],      # F1 high
            -x[1]['metrics']['accuracy'],      # Accuracy high
            x[1]['training_time']              # Time low
        )
    )
    
    # Best overall (first in sorted list)
    best_overall_key, best_overall = results_sorted[0]
    st.success(f"""
        ** Best Overall Performance**: {best_overall['display_name']}
        - F1-Score: {best_overall['metrics']['f1_score']:.3f}
        - Accuracy: {best_overall['metrics']['accuracy']*100:.2f}%
        - Training Time: {best_overall['training_time']:.2f}s
        - Good balance of precision and recall
    """)
    
    # Fastest
    fastest = min(results.values(), key=lambda x: x['training_time'])
    if fastest['display_name'] != best_overall['display_name']:
        st.info(f"""
            ** Fastest Training**: {fastest['display_name']}
            - Training Time: {fastest['training_time']:.2f}s
            - Use if speed is critical for your application
        """)
    
    # Most Accurate (only show if different from best overall)
    best_acc = max(results.values(), key=lambda x: x['metrics']['accuracy'])
    if best_acc['display_name'] != best_overall['display_name']:
        st.success(f"""
            ** Highest Accuracy**: {best_acc['display_name']}
            - Accuracy: {best_acc['metrics']['accuracy']*100:.2f}%
        """)

    
    # Mark comparison as completed
    st.session_state.comparison_completed = True
    
    # Navigation
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button(" Back to Training", use_container_width=True):
            st.session_state.current_step = 5
            st.rerun()
    with col3:
        if st.button("Continue to Report ", use_container_width=True, type="primary"):
            st.session_state.current_step = 7
            st.rerun()
