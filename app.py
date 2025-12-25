"""
AutoML Classification System - Main Application
A comprehensive web-based AutoML system for supervised classification tasks.
"""

import streamlit as st
import sys
import os

# Add modules to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import modules
from modules.data_upload import run_module_1

# Page configuration
st.set_page_config(
    page_title="AutoML Classification System",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Dark Green Theme CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
    
    /* === Global Styles - Dark Green Theme === */
    :root {
        --primary: #22c55e;
        --primary-light: #4ade80;
        --primary-dark: #16a34a;
        --bg-dark: #0a0f0a;
        --bg-card: #111a11;
        --bg-secondary: #0d140d;
        --text-primary: #e8f5e9;
        --text-secondary: #9ca3af;
        --border-color: rgba(34, 197, 94, 0.2);
        --shadow-green: 0 0 20px rgba(34, 197, 94, 0.15);
    }
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    /* === Main Container === */
    .main {
        background: var(--bg-dark) !important;
        padding: 1rem 2rem;
    }
    
    .stApp {
        background: var(--bg-dark) !important;
    }
    
    .block-container {
        padding: 2rem 1rem;
        max-width: 1400px;
        background: var(--bg-dark) !important;
    }
    
    /* === Headings === */
    h1, h2, h3, h4 {
        color: var(--text-primary) !important;
        -webkit-text-fill-color: var(--text-primary) !important;
    }
    
    h1 {
        font-weight: 800 !important;
        font-size: 2.5rem !important;
        letter-spacing: -0.02em;
        margin-bottom: 1.5rem !important;
        padding-bottom: 1rem;
        border-bottom: 2px solid var(--primary);
    }
    
    h2 {
        font-weight: 700 !important;
        font-size: 1.75rem !important;
        margin-top: 2rem !important;
        margin-bottom: 1rem !important;
        color: var(--primary) !important;
        -webkit-text-fill-color: var(--primary) !important;
    }
    
    h3 {
        font-weight: 600 !important;
        font-size: 1.25rem !important;
        color: var(--primary-light) !important;
        -webkit-text-fill-color: var(--primary-light) !important;
    }
    
    p, span, label, div {
        color: var(--text-primary) !important;
    }
    
    /* === Metrics Cards === */
    [data-testid="stMetric"] {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color);
        padding: 1.25rem !important;
        border-radius: 12px;
        box-shadow: var(--shadow-green);
        transition: all 0.3s ease;
    }
    
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-color: var(--primary);
        box-shadow: 0 0 30px rgba(34, 197, 94, 0.25);
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.8rem !important;
        color: var(--text-secondary) !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 1.75rem !important;
        font-weight: 700 !important;
        color: var(--primary) !important;
    }
    
    /* === Buttons === */
    .stButton > button {
        background: #16a34a !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: #15803d !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 20px rgba(22, 163, 74, 0.4);
    }

    
    .stButton > button[kind="secondary"] {
        background: transparent !important;
        border: 1px solid var(--primary) !important;
        color: var(--primary) !important;
    }
    
    .stButton > button[kind="secondary"]:hover {
        background: rgba(34, 197, 94, 0.1) !important;
    }
    
    /* === DataFrame === */
    [data-testid="stDataFrame"] {
        border-radius: 12px !important;
        overflow: hidden;
        border: 1px solid var(--border-color);
        background: var(--bg-card) !important;
    }
    
    [data-testid="stDataFrame"] table {
        background: var(--bg-card) !important;
    }
    
    [data-testid="stDataFrame"] thead tr th {
        background: var(--primary) !important;
        color: #0a0f0a !important;
        font-weight: 600 !important;
        padding: 0.75rem !important;
        text-transform: uppercase;
        font-size: 0.7rem;
        letter-spacing: 0.05em;
    }
    
    [data-testid="stDataFrame"] tbody tr {
        background: var(--bg-card) !important;
    }
    
    [data-testid="stDataFrame"] tbody tr:hover {
        background: rgba(34, 197, 94, 0.1) !important;
    }
    
    /* === File Uploader === */
    [data-testid="stFileUploader"] {
        background: var(--bg-card) !important;
        border: 2px dashed var(--border-color);
        border-radius: 12px;
        padding: 2rem;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: var(--primary);
        background: rgba(34, 197, 94, 0.05) !important;
    }
    
    /* === Alerts === */
    .stAlert {
        border-radius: 12px !important;
        padding: 1rem 1.25rem !important;
        background: var(--bg-card) !important;
        border: none !important;
    }
    
    .stSuccess {
        background: rgba(34, 197, 94, 0.1) !important;
        border-left: 4px solid var(--primary) !important;
    }
    
    .stWarning {
        background: rgba(234, 179, 8, 0.1) !important;
        border-left: 4px solid #eab308 !important;
    }
    
    .stError {
        background: rgba(239, 68, 68, 0.1) !important;
        border-left: 4px solid #ef4444 !important;
    }
    
    .stInfo {
        background: rgba(59, 130, 246, 0.1) !important;
        border-left: 4px solid #3b82f6 !important;
    }
    
    /* === Sidebar === */
    [data-testid="stSidebar"] {
        background: var(--bg-secondary) !important;
        border-right: 1px solid var(--border-color);
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: var(--text-primary);
    }
    
    [data-testid="stSidebar"] hr {
        border-color: var(--border-color);
    }
    
    /* === Selectbox & Inputs === */
    .stSelectbox > div > div,
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        color: var(--text-primary) !important;
    }
    
    .stSelectbox > div > div:hover,
    .stTextInput > div > div > input:hover,
    .stNumberInput > div > div > input:hover {
        border-color: var(--primary) !important;
    }
    
    /* === Tabs === */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: var(--bg-card);
        padding: 0.25rem;
        border-radius: 10px;
        border: 1px solid var(--border-color);
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.5rem 1rem;
        color: var(--text-secondary) !important;
        background: transparent !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(34, 197, 94, 0.1) !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--primary) !important;
        color: #0a0f0a !important;
    }
    
    /* === Expander === */
    .streamlit-expanderHeader {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 10px !important;
        color: var(--text-primary) !important;
    }
    
    .streamlit-expanderHeader:hover {
        border-color: var(--primary) !important;
    }
    
    .streamlit-expanderContent {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-top: none !important;
        border-radius: 0 0 10px 10px !important;
    }
    
    /* === Divider === */
    hr {
        border: none;
        height: 1px;
        background: var(--border-color);
        margin: 2rem 0;
    }
    
    /* === Progress Bar === */
    .stProgress > div > div {
        background: var(--primary) !important;
        border-radius: 8px;
    }
    
    /* === Scrollbar === */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-dark);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--primary);
        border-radius: 8px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--primary-light);
    }
    
    /* === Plotly Charts === */
    .js-plotly-plot {
        border-radius: 12px;
        border: 1px solid var(--border-color);
        overflow: hidden;
    }
    
    /* === Checkbox & Radio === */
    .stCheckbox, .stRadio {
        color: var(--text-primary) !important;
    }
    
    /* === Download Button === */
    .stDownloadButton > button {
        background: var(--bg-card) !important;
        border: 1px solid var(--primary) !important;
        color: var(--primary) !important;
    }
    
    .stDownloadButton > button:hover {
        background: var(--primary) !important;
        color: #0a0f0a !important;
    }
    
    /* === Slider === */
    .stSlider > div > div > div {
        background: var(--primary) !important;
    }
    
    /* === Multiselect === */
    .stMultiSelect > div {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
    }
    </style>
""", unsafe_allow_html=True)



def initialize_session_state():
    """Initialize session state variables."""
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 1
    
    if 'df' not in st.session_state:
        st.session_state.df = None
    
    if 'target_column' not in st.session_state:
        st.session_state.target_column = None
    
    if 'preprocessing_decisions' not in st.session_state:
        st.session_state.preprocessing_decisions = {}
    
    if 'trained_models' not in st.session_state:
        st.session_state.trained_models = {}
    
    if 'eda_results' not in st.session_state:
        st.session_state.eda_results = {}


def display_sidebar():
    """Display modern professional sidebar with navigation and progress."""
    with st.sidebar:
        # Modern Logo/Title Section - Dark Green Theme
        st.markdown("""
            <div style="text-align: center; padding: 1.5rem 0; margin-bottom: 2rem;">
                <div style="background: #22c55e;
                            width: 50px; height: 50px; border-radius: 12px; margin: 0 auto 1rem auto;
                            display: flex; align-items: center; justify-content: center;
                            box-shadow: 0 8px 24px rgba(34, 197, 94, 0.3);">
                    <span style="font-size: 1.5rem; color: #0a0f0a; font-weight: 800;">ML</span>
                </div>
                <h2 style="color: #e8f5e9; margin: 0; font-size: 1.25rem; font-weight: 700;">AutoML</h2>
                <p style="color: rgba(232, 245, 233, 0.6); margin: 0.25rem 0 0 0; font-size: 0.75rem;">Classification System</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Progress Section
        st.markdown("""
            <div style="margin-bottom: 1.5rem;">
                <h3 style="color: #22c55e; font-size: 0.75rem; 
                           text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 1rem; font-weight: 600;">
                    Pipeline Progress
                </h3>
            </div>
        """, unsafe_allow_html=True)
        
        steps = [
            ("Dataset Upload", 1),
            ("Data Analysis", 2),
            ("Issue Detection", 3),
            ("Preprocessing", 4),
            ("Model Training", 5),
            ("Comparison", 6),
            ("Final Report", 7)
        ]
        
        current_step = st.session_state.current_step
        progress_percentage = (current_step / 7) * 100
        
        # Progress bar - Dark green theme
        st.markdown(f"""
            <div style="background: rgba(34, 197, 94, 0.1); height: 6px; border-radius: 8px; overflow: hidden; margin-bottom: 1.5rem;">
                <div style="background: #22c55e; 
                            width: {progress_percentage}%; height: 100%; border-radius: 8px; 
                            transition: width 0.3s ease;"></div>
            </div>
        """, unsafe_allow_html=True)
        
        # Determine which steps can be accessed
        # Module 1 is always accessible
        # Module 2+ require dataset to be loaded
        dataset_loaded = 'df' in st.session_state and st.session_state.df is not None
        
        # Step list with modern design and clickable navigation
        for step_name, step_num in steps:
            # Determine status and styling - Dark green theme
            if step_num < current_step:
                # Completed - always clickable to go back
                status_icon = ""
                status_color = "#22c55e"
                bg_color = "rgba(34, 197, 94, 0.1)"
                border_color = "rgba(34, 197, 94, 0.3)"
                text_color = "#e8f5e9"
                clickable = True  # Always allow going back to completed steps
            elif step_num == current_step:
                # Current - always clickable
                status_icon = ""
                status_color = "#22c55e"
                bg_color = "rgba(34, 197, 94, 0.15)"
                border_color = "#22c55e"
                text_color = "#e8f5e9"
                clickable = True
            else:
                # Future steps - check accessibility
                status_icon = ""
                status_color = "rgba(232, 245, 233, 0.3)"
                bg_color = "rgba(34, 197, 94, 0.03)"
                border_color = "rgba(34, 197, 94, 0.1)"
                text_color = "rgba(232, 245, 233, 0.5)"
                
                # Module 1 is always accessible
                if step_num == 1:
                    clickable = True
                # Module 2 accessible if dataset is loaded
                elif step_num == 2 and dataset_loaded:
                    clickable = True
                # Module 3 accessible if EDA completed or current step is past it
                elif step_num == 3 and st.session_state.get('eda_completed', False):
                    clickable = True
                # Module 4 accessible if issues completed and preprocessing not skipped
                elif step_num == 4 and st.session_state.get('issues_completed', False):
                    if not st.session_state.get('skip_preprocessing', False):
                        clickable = True
                    else:
                        clickable = False
                # Module 5 accessible if preprocessing done OR skipped
                elif step_num == 5 and (st.session_state.get('preprocessing_completed', False) or st.session_state.get('skip_preprocessing', False)):
                    clickable = True
                # Module 6 accessible if training completed
                elif step_num == 6 and st.session_state.get('training_completed', False):
                    clickable = True
                # Module 7 accessible if training completed
                elif step_num == 7 and st.session_state.get('training_completed', False):
                    clickable = True
                else:
                    clickable = False

            
            # Create clickable or non-clickable step
            if clickable:
                # Use button for clickable steps
                button_key = f"nav_step_{step_num}"
                if st.button(
                    f"{status_icon} {step_name}",
                    key=button_key,
                    use_container_width=True,
                    type="secondary" if step_num == current_step else "tertiary"
                ):
                    if step_num != current_step:
                        st.session_state.current_step = step_num
                        st.rerun()
            else:
                # Display as non-clickable for locked steps
                st.markdown(f"""
                    <div style="background: {bg_color}; 
                                border: 1px solid {border_color}; 
                                border-radius: 10px; 
                                padding: 0.75rem 1rem; 
                                margin-bottom: 0.5rem;
                                display: flex;
                                align-items: center;
                                opacity: 0.6;
                                cursor: not-allowed;">
                        <span style="color: {status_color}; 
                                     font-size: 0.875rem; 
                                     font-weight: 700; 
                                     margin-right: 0.75rem;
                                     width: 20px;
                                     text-align: center;">{status_icon}</span>
                        <span style="color: {text_color}; 
                                     font-size: 0.875rem; 
                                     font-weight: 500;">{step_name}</span>
                    </div>
                """, unsafe_allow_html=True)
        
        st.markdown("<div style='margin: 2rem 0;'></div>", unsafe_allow_html=True)
        
        # Dataset info card (if loaded)
        if st.session_state.df is not None:
            st.markdown("""
                <div style="margin-bottom: 1.5rem;">
                    <h3 style="color: rgba(255,255,255,0.9); font-size: 0.875rem; 
                               text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 1rem; font-weight: 600;">
                        Dataset Info
                    </h3>
                </div>
            """, unsafe_allow_html=True)
            
            df_info = [
                ("Rows", f"{len(st.session_state.df):,}"),
                ("Columns", f"{len(st.session_state.df.columns)}"),
            ]
            
            if st.session_state.target_column:
                df_info.append(("Target", st.session_state.target_column))
            
            for label, value in df_info:
                st.markdown(f"""
                    <div style="background: rgba(255,255,255,0.05); 
                                border-radius: 8px; 
                                padding: 0.75rem 1rem; 
                                margin-bottom: 0.5rem;
                                display: flex;
                                justify-content: space-between;
                                align-items: center;">
                        <span style="color: rgba(255,255,255,0.6); font-size: 0.8125rem; font-weight: 500;">{label}</span>
                        <span style="color: white; font-size: 0.875rem; font-weight: 600;">{value}</span>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<div style='margin: 2rem 0;'></div>", unsafe_allow_html=True)
        
        # Reset button with modern styling
        if st.button(" Start Over", type="secondary", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        
        # Footer
        st.markdown("""
            <div style="margin-top: 3rem; padding-top: 1.5rem; border-top: 1px solid rgba(255,255,255,0.1); text-align: center;">
                <p style="color: rgba(255,255,255,0.4); font-size: 0.75rem; margin: 0;">
                    Powered by Streamlit
                </p>
            </div>
        """, unsafe_allow_html=True)


def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()
    
    # Display sidebar
    display_sidebar()
    
    # Main content area
    current_step = st.session_state.current_step
    
    if current_step == 1:
        # Module 1: Dataset Upload & Basic Info
        from modules.data_upload import run_module_1
        df, target_column = run_module_1()
        
        if df is not None and target_column is not None:
            # Store in session state
            st.session_state.df = df
            st.session_state.target_column = target_column
            
            # Add button to proceed to EDA
            st.markdown("---")
            if st.button("Continue to EDA ", type="primary", use_container_width=True):
                st.session_state.current_step = 2
                st.rerun()
    
    elif current_step == 2:
        # Module 2: Automated EDA
        if st.session_state.df is not None and st.session_state.target_column is not None:
            from modules.eda import run_module_2
            run_module_2(st.session_state.df, st.session_state.target_column)
        else:
            st.error(" No dataset loaded. Please upload a dataset first.")
            if st.button(" Back to Upload"):
                st.session_state.current_step = 1
                st.rerun()
    
    elif current_step == 3:
        # Module 3: Issue Detection & User Approval
        if st.session_state.df is not None and st.session_state.target_column is not None:
            from modules.issue_detection import run_module_3
            run_module_3(st.session_state.df, st.session_state.target_column)
        else:
            st.error(" No dataset loaded. Please upload a dataset first.")
            if st.button(" Back to Upload"):
                st.session_state.current_step = 1
                st.rerun()
    
    elif current_step == 4:
        # Module 4: Preprocessing
        if st.session_state.df is not None and st.session_state.target_column is not None:
            from modules.preprocessing import run_module_4
            run_module_4(st.session_state.df, st.session_state.target_column)
        else:
            st.error(" No dataset loaded. Please upload a dataset first.")
            if st.button(" Back to Upload"):
                st.session_state.current_step = 1
                st.rerun()
    
    elif current_step == 5:
        # Module 5: Model Training & Evaluation
        from modules.model_training import run_module_5
        run_module_5()
    
    elif current_step == 6:
        # Module 6: Model Comparison Dashboard
        from modules.model_comparison import run_module_6
        run_module_6()
    
    elif current_step == 7:
        # Module 7: Auto-Generated Final Report
        from modules.report_generation import run_module_7
        run_module_7()
    
    else:
        st.info(f" Module {current_step} - Under Development")


if __name__ == "__main__":
    main()
