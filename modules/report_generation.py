"""
Module 7: Auto-Generated Final Report

Comprehensive PDF report with dark theme, graphs, and professional layout.
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List
from datetime import datetime
import io


def create_class_distribution_chart(class_labels: List, class_counts: Dict) -> bytes:
    """Create class distribution bar chart with dark theme."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(5, 2.5))
    fig.patch.set_facecolor('#0a0f0a')
    ax.set_facecolor('#0a0f0a')
    
    labels = [str(c) for c in class_labels]
    counts = [class_counts.get(c, 0) for c in class_labels]
    colors = ['#22c55e', '#16a34a', '#15803d', '#166534', '#14532d'][:len(labels)]
    
    bars = ax.bar(labels, counts, color=colors, edgecolor='#22c55e', linewidth=1)
    ax.set_xlabel('Class', fontsize=9, color='#e8f5e9', fontweight='bold')
    ax.set_ylabel('Count', fontsize=9, color='#e8f5e9', fontweight='bold')
    ax.set_title('Target Class Distribution', fontsize=10, fontweight='bold', color='#22c55e')
    
    ax.tick_params(colors='#e8f5e9', labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#22c55e')
    ax.spines['left'].set_color('#22c55e')
    
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.02, 
                f'{count:,}', ha='center', va='bottom', fontsize=8, color='#e8f5e9', fontweight='bold')
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#0a0f0a')
    plt.close()
    buf.seek(0)
    return buf.getvalue()


def create_missing_values_chart(missing_info: Dict, df_len: int) -> bytes:
    """Create missing values bar chart with dark theme."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(5, 2.5))
    fig.patch.set_facecolor('#0a0f0a')
    ax.set_facecolor('#0a0f0a')
    
    if not missing_info:
        ax.text(0.5, 0.5, 'No Missing Values', ha='center', va='center', 
                fontsize=12, fontweight='bold', color='#22c55e')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    else:
        sorted_missing = sorted(missing_info.items(), key=lambda x: x[1], reverse=True)[:8]
        columns = [c[:12] for c, v in sorted_missing]
        percentages = [(v / df_len) * 100 for c, v in sorted_missing]
        
        bars = ax.barh(columns, percentages, color='#ef4444', edgecolor='#fca5a5', linewidth=0.5)
        ax.set_xlabel('Missing %', fontsize=9, color='#e8f5e9', fontweight='bold')
        ax.set_title('Missing Values by Column', fontsize=10, fontweight='bold', color='#22c55e')
        ax.tick_params(colors='#e8f5e9', labelsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#22c55e')
        ax.spines['left'].set_color('#22c55e')
        
        for bar, pct in zip(bars, percentages):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2, 
                    f'{pct:.1f}%', ha='left', va='center', fontsize=7, color='#e8f5e9')
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#0a0f0a')
    plt.close()
    buf.seek(0)
    return buf.getvalue()


def create_correlation_matrix(df: pd.DataFrame) -> bytes:
    """Create correlation matrix heatmap with dark theme."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fig, ax = plt.subplots(figsize=(5, 3.5))
    fig.patch.set_facecolor('#0a0f0a')
    ax.set_facecolor('#0a0f0a')
    
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:8]
    
    if len(num_cols) > 1:
        corr = df[num_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        cmap = sns.color_palette("Greens", as_cmap=True)
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap=cmap,
                    center=0, square=True, linewidths=0.5, ax=ax,
                    annot_kws={'size': 7, 'color': '#0a0f0a'}, 
                    cbar_kws={'shrink': 0.7})
        ax.set_title('Correlation Matrix', fontsize=10, fontweight='bold', color='#22c55e')
        ax.tick_params(colors='#e8f5e9', labelsize=7)
    else:
        ax.text(0.5, 0.5, 'Not enough numerical features', ha='center', va='center', 
                fontsize=10, color='#e8f5e9')
        ax.axis('off')
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#0a0f0a')
    plt.close()
    buf.seek(0)
    return buf.getvalue()


def create_feature_distributions(df: pd.DataFrame) -> bytes:
    """Create feature distribution histograms with dark theme."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:4]
    
    if not num_cols:
        fig, ax = plt.subplots(figsize=(5, 2))
        fig.patch.set_facecolor('#0a0f0a')
        ax.set_facecolor('#0a0f0a')
        ax.text(0.5, 0.5, 'No numerical features', ha='center', va='center', 
                fontsize=10, color='#e8f5e9')
        ax.axis('off')
    else:
        n_cols = min(2, len(num_cols))
        n_rows = (len(num_cols) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5, 1.8*n_rows))
        fig.patch.set_facecolor('#0a0f0a')
        
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        
        for i, col in enumerate(num_cols):
            ax = axes[i]
            ax.set_facecolor('#0a0f0a')
            data = df[col].dropna()
            ax.hist(data, bins=15, color='#22c55e', edgecolor='#16a34a', linewidth=0.5, alpha=0.9)
            ax.set_title(col[:12], fontsize=8, fontweight='bold', color='#e8f5e9')
            ax.tick_params(colors='#e8f5e9', labelsize=6)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color('#22c55e')
            ax.spines['left'].set_color('#22c55e')
        
        for i in range(len(num_cols), len(axes)):
            axes[i].axis('off')
            axes[i].set_facecolor('#0a0f0a')
    
    plt.suptitle('Feature Distributions', fontsize=10, fontweight='bold', color='#22c55e', y=1.02)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#0a0f0a')
    plt.close()
    buf.seek(0)
    return buf.getvalue()


def generate_pdf_report(
    df: pd.DataFrame,
    target_column: str,
    eda_results: Dict,
    user_approvals: Dict,
    preprocessed_data: Dict,
    trained_models: Dict,
    class_labels: List,
    skip_preprocessing: bool = False
) -> bytes:
    """Generate dark-themed PDF report with proper formatting."""
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    
    # Dark theme colors
    BG_DARK = colors.HexColor('#0a0f0a')
    BG_CARD = colors.HexColor('#111a11')
    GREEN_PRIMARY = colors.HexColor('#22c55e')
    GREEN_DARK = colors.HexColor('#16a34a')
    TEXT_PRIMARY = colors.HexColor('#e8f5e9')
    TEXT_SECONDARY = colors.HexColor('#9ca3af')
    BORDER_COLOR = colors.HexColor('#22c55e')
    
    buffer = io.BytesIO()
    
    # Custom page with dark background and border
    def draw_page_background(canvas, doc):
        canvas.saveState()
        # Dark background
        canvas.setFillColor(BG_DARK)
        canvas.rect(0, 0, letter[0], letter[1], fill=1, stroke=0)
        # Green border
        canvas.setStrokeColor(GREEN_PRIMARY)
        canvas.setLineWidth(2)
        canvas.rect(20, 20, letter[0]-40, letter[1]-40, fill=0, stroke=1)
        # Inner border line
        canvas.setStrokeColor(GREEN_DARK)
        canvas.setLineWidth(0.5)
        canvas.rect(25, 25, letter[0]-50, letter[1]-50, fill=0, stroke=1)
        canvas.restoreState()
    
    doc = SimpleDocTemplate(
        buffer, pagesize=letter,
        topMargin=0.5*inch, bottomMargin=0.4*inch,
        leftMargin=0.6*inch, rightMargin=0.6*inch
    )
    
    elements = []
    
    # Styles for dark theme
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('Title', fontSize=18, textColor=GREEN_PRIMARY, 
                                  spaceAfter=8, alignment=TA_CENTER, fontName='Helvetica-Bold')
    section_style = ParagraphStyle('Section', fontSize=12, textColor=GREEN_PRIMARY, 
                                    spaceAfter=6, spaceBefore=8, fontName='Helvetica-Bold')
    subsection_style = ParagraphStyle('Subsection', fontSize=10, textColor=TEXT_PRIMARY, 
                                       spaceAfter=4, spaceBefore=6, fontName='Helvetica-Bold')
    normal_style = ParagraphStyle('Normal', fontSize=8, textColor=TEXT_PRIMARY, 
                                   spaceAfter=2, leading=10)
    small_style = ParagraphStyle('Small', fontSize=7, textColor=TEXT_SECONDARY, 
                                  spaceAfter=2, leading=9)
    
    # Get best model
    results_sorted = sorted(
        trained_models.items(),
        key=lambda x: (-x[1]['metrics']['f1_score'], -x[1]['metrics']['accuracy'], x[1]['training_time'])
    )
    best_model_key, best_model = results_sorted[0]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Dark table style
    dark_table_style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), GREEN_DARK),
        ('TEXTCOLOR', (0, 0), (-1, 0), BG_DARK),
        ('BACKGROUND', (0, 1), (-1, -1), BG_CARD),
        ('TEXTCOLOR', (0, 1), (-1, -1), TEXT_PRIMARY),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, GREEN_DARK),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ])
    
    # ===== TITLE =====
    elements.append(Paragraph("AutoML Classification Report", title_style))
    elements.append(Paragraph(f"Generated: {timestamp}", small_style))
    elements.append(Spacer(1, 0.15*inch))
    
    # ===== EXECUTIVE SUMMARY =====
    elements.append(Paragraph("Executive Summary", section_style))
    exec_data = [
        ['Metric', 'Value'],
        ['Dataset', f'{len(df):,} rows x {len(df.columns)} cols'],
        ['Target', target_column],
        ['Classes', str(len(class_labels))],
        ['Best Model', best_model['display_name']],
        ['F1-Score', f"{best_model['metrics']['f1_score']:.4f}"],
        ['Accuracy', f"{best_model['metrics']['accuracy']*100:.2f}%"],
    ]
    exec_table = Table(exec_data, colWidths=[1.8*inch, 4.5*inch])
    exec_table.setStyle(dark_table_style)
    elements.append(exec_table)
    elements.append(Spacer(1, 0.15*inch))
    
    # ===== 1. DATASET OVERVIEW =====
    elements.append(Paragraph("1. Dataset Overview", section_style))
    col_data = [['Column', 'Type', 'Non-Null', 'Missing%']]
    for col in df.columns[:10]:
        dtype = str(df[col].dtype)[:8]
        non_null = df[col].count()
        missing_pct = (df[col].isna().sum() / len(df)) * 100
        col_data.append([col[:18], dtype, f'{non_null:,}', f'{missing_pct:.1f}%'])
    if len(df.columns) > 10:
        col_data.append([f'+{len(df.columns)-10} more', '', '', ''])
    
    col_table = Table(col_data, colWidths=[2.2*inch, 1.2*inch, 1.2*inch, 1*inch])
    col_table.setStyle(dark_table_style)
    elements.append(col_table)
    elements.append(Spacer(1, 0.15*inch))
    
    # ===== 2. EDA FINDINGS =====
    elements.append(Paragraph("2. EDA Findings", section_style))
    
    num_features = len(eda_results.get('numerical_features', []))
    cat_features = len(eda_results.get('categorical_features', []))
    missing_info = eda_results.get('missing_values', {})
    total_missing = sum(missing_info.values()) if missing_info else 0
    
    eda_text = f"Features: {num_features} numerical, {cat_features} categorical | Missing: {total_missing:,} values"
    elements.append(Paragraph(eda_text, normal_style))
    elements.append(Spacer(1, 0.08*inch))
    
    # Graphs side by side
    class_counts = df[target_column].value_counts().to_dict()
    class_chart = create_class_distribution_chart(class_labels, class_counts)
    missing_chart = create_missing_values_chart(missing_info, len(df))
    
    img1 = Image(io.BytesIO(class_chart), width=3.2*inch, height=1.6*inch)
    img2 = Image(io.BytesIO(missing_chart), width=3.2*inch, height=1.6*inch)
    graph_table = Table([[img1, img2]], colWidths=[3.3*inch, 3.3*inch])
    elements.append(graph_table)
    elements.append(Spacer(1, 0.1*inch))
    
    # Correlation and distributions
    corr_chart = create_correlation_matrix(df)
    dist_chart = create_feature_distributions(df)
    
    img3 = Image(io.BytesIO(corr_chart), width=3.2*inch, height=2.2*inch)
    img4 = Image(io.BytesIO(dist_chart), width=3.2*inch, height=2.2*inch)
    graph_table2 = Table([[img3, img4]], colWidths=[3.3*inch, 3.3*inch])
    elements.append(graph_table2)
    elements.append(Spacer(1, 0.15*inch))
    
    # ===== 3. DETECTED ISSUES =====
    elements.append(Paragraph("3. Detected Issues", section_style))
    if user_approvals:
        issues_data = [['Issue', 'Column', 'Action']]
        for col, approval in list(user_approvals.items())[:8]:
            issue_type = approval.get('issue_type', 'Unknown')[:12]
            action = approval.get('action', 'N/A')[:10]
            issues_data.append([issue_type.title(), col[:15], action.title()])
        issues_table = Table(issues_data, colWidths=[2*inch, 2.2*inch, 1.8*inch])
        issues_table.setStyle(dark_table_style)
        elements.append(issues_table)
    else:
        elements.append(Paragraph("No data quality issues detected.", normal_style))
    elements.append(Spacer(1, 0.12*inch))
    
    # ===== 4. PREPROCESSING =====
    elements.append(Paragraph("4. Preprocessing Decisions", section_style))
    if skip_preprocessing:
        elements.append(Paragraph("Preprocessing SKIPPED - dataset was clean.", normal_style))
    elif preprocessed_data and 'X_train' in preprocessed_data:
        prep_text = f"Train: {len(preprocessed_data['X_train']):,} | Test: {len(preprocessed_data['X_test']):,}"
        if 'feature_names' in preprocessed_data:
            prep_text += f" | Features: {len(preprocessed_data['feature_names'])}"
        elements.append(Paragraph(prep_text, normal_style))
    else:
        elements.append(Paragraph("Basic preprocessing applied.", normal_style))
    elements.append(Spacer(1, 0.12*inch))
    
    # ===== 5. MODEL CONFIGURATIONS =====
    elements.append(Paragraph("5. Model Configurations", section_style))
    config_data = [['Model', 'Parameters', 'Time']]
    for model_key, result in results_sorted[:7]:
        params = result.get('best_params', {})
        params_str = ', '.join([f"{k}={v}" for k,v in list(params.items())[:2]]) if params else 'Default'
        config_data.append([result['display_name'][:16], params_str[:30], f"{result['training_time']:.2f}s"])
    
    config_table = Table(config_data, colWidths=[2*inch, 3*inch, 1*inch])
    config_table.setStyle(dark_table_style)
    elements.append(config_table)
    elements.append(Spacer(1, 0.12*inch))
    
    # ===== 6. MODEL COMPARISON =====
    elements.append(Paragraph("6. Model Comparison", section_style))
    comp_data = [['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']]
    for model_key, result in results_sorted:
        m = result['metrics']
        comp_data.append([
            result['display_name'][:14],
            f"{m['accuracy']*100:.1f}%",
            f"{m['precision']:.3f}",
            f"{m['recall']:.3f}",
            f"{m['f1_score']:.3f}"
        ])
    
    comp_table = Table(comp_data, colWidths=[1.8*inch, 1.1*inch, 1.1*inch, 1*inch, 1*inch])
    # Highlight best model row
    comp_style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), GREEN_DARK),
        ('TEXTCOLOR', (0, 0), (-1, 0), BG_DARK),
        ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#1a3a1a')),  # Highlight best
        ('BACKGROUND', (0, 2), (-1, -1), BG_CARD),
        ('TEXTCOLOR', (0, 1), (-1, -1), TEXT_PRIMARY),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, GREEN_DARK),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
    ])
    comp_table.setStyle(comp_style)
    elements.append(comp_table)
    elements.append(Spacer(1, 0.12*inch))
    
    # ===== 7. BEST MODEL SUMMARY =====
    elements.append(Paragraph("7. Best Model Summary", section_style))
    elements.append(Paragraph(f"Selected: {best_model['display_name']}", subsection_style))
    
    # Metrics table
    metrics_data = [
        ['Metric', 'Value'],
        ['Accuracy', f"{best_model['metrics']['accuracy']*100:.2f}%"],
        ['Precision', f"{best_model['metrics']['precision']:.4f}"],
        ['Recall', f"{best_model['metrics']['recall']:.4f}"],
        ['F1-Score', f"{best_model['metrics']['f1_score']:.4f}"],
        ['Training Time', f"{best_model['training_time']:.2f}s"],
    ]
    if best_model['metrics'].get('roc_auc'):
        metrics_data.append(['ROC-AUC', f"{best_model['metrics']['roc_auc']:.4f}"])
    
    metrics_table = Table(metrics_data, colWidths=[2*inch, 2*inch])
    metrics_table.setStyle(dark_table_style)
    elements.append(metrics_table)
    elements.append(Spacer(1, 0.08*inch))
    
    # Confusion Matrix
    elements.append(Paragraph("Confusion Matrix:", normal_style))
    cm = best_model['metrics']['confusion_matrix']
    cm_header = [''] + [str(c)[:8] for c in class_labels]
    cm_data = [cm_header]
    for i, label in enumerate(class_labels):
        row = [str(label)[:8]] + [str(cm[i][j]) for j in range(len(cm))]
        cm_data.append(row)
    
    cm_table = Table(cm_data, colWidths=[1*inch] + [0.7*inch]*len(class_labels))
    cm_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), GREEN_DARK),
        ('BACKGROUND', (0, 1), (0, -1), GREEN_DARK),
        ('TEXTCOLOR', (0, 0), (-1, 0), BG_DARK),
        ('TEXTCOLOR', (0, 1), (0, -1), BG_DARK),
        ('BACKGROUND', (1, 1), (-1, -1), BG_CARD),
        ('TEXTCOLOR', (1, 1), (-1, -1), TEXT_PRIMARY),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, GREEN_DARK),
        ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
    ]))
    elements.append(cm_table)
    elements.append(Spacer(1, 0.08*inch))
    
    # Justification
    elements.append(Paragraph("Selection Justification:", normal_style))
    elements.append(Paragraph(f"- Highest F1-Score: {best_model['metrics']['f1_score']:.4f}", small_style))
    elements.append(Paragraph(f"- Strong Accuracy: {best_model['metrics']['accuracy']*100:.2f}%", small_style))
    elements.append(Paragraph(f"- Efficient: {best_model['training_time']:.2f}s training time", small_style))
    
    # Build PDF with dark background
    doc.build(elements, onFirstPage=draw_page_background, onLaterPages=draw_page_background)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    
    return pdf_bytes


# ==================== UI FUNCTION ====================

def run_module_7() -> None:
    """Main function for Module 7: Auto-Generated Final Report."""
    st.markdown("""
        <div style="text-align: center; padding: 1.5rem 0;">
            <h1 style="font-size: 2.5rem; margin-bottom: 0.5rem;">Generate Final Report</h1>
            <p style="font-size: 1rem; color: #9ca3af;">
                Dark-themed PDF report with graphs
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    if not st.session_state.get('training_completed', False):
        st.error("Please complete model training before generating the report.")
        if st.button("Go to Training"):
            st.session_state.current_step = 5
            st.rerun()
        return
    
    df = st.session_state.df
    target_column = st.session_state.target_column
    eda_results = st.session_state.get('eda_results', {})
    user_approvals = st.session_state.get('user_approvals', {})
    preprocessed_data = st.session_state.get('preprocessed_data', {})
    trained_models = st.session_state.trained_models
    class_labels = st.session_state.get('class_labels', [])
    skip_preprocessing = st.session_state.get('skip_preprocessing', False)
    
    st.markdown("## Report Contents")
    st.info("""
    **Dark-themed PDF with:**
    - Executive Summary
    - Dataset Overview
    - EDA with 4 Charts
    - Detected Issues
    - Preprocessing Steps
    - Model Configurations
    - Model Comparison
    - Best Model Summary
    """)
    
    st.markdown("---")
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    try:
        with st.spinner("Generating dark-themed PDF..."):
            pdf_bytes = generate_pdf_report(
                df=df, target_column=target_column, eda_results=eda_results,
                user_approvals=user_approvals, preprocessed_data=preprocessed_data,
                trained_models=trained_models, class_labels=class_labels,
                skip_preprocessing=skip_preprocessing
            )
        
        st.download_button(
            label="Download PDF Report",
            data=pdf_bytes,
            file_name=f"AutoML_Report_{timestamp_str}.pdf",
            mime="application/pdf",
            use_container_width=True,
            type="primary"
        )
        st.success("Report ready!")
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.exception(e)
    
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("Back to Comparison", use_container_width=True):
            st.session_state.current_step = 6
            st.rerun()
    with col3:
        if st.button("Finish", use_container_width=True, type="primary"):
            st.balloons()
            st.success("AutoML Pipeline Complete!")
