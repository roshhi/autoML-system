# AutoML Classification System

A comprehensive, web-based automated machine learning platform designed for supervised classification tasks. This system provides an intuitive, no-code interface that guides users through the complete ML pipeline—from data upload to final model deployment.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Technology Stack](#technology-stack)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Project Structure](#project-structure)
7. [Pipeline Modules](#pipeline-modules)
8. [Supported Algorithms](#supported-algorithms)
9. [Evaluation Metrics](#evaluation-metrics)
10. [Screenshots](#screenshots)
11. [Contributors](#contributors)
12. [License](#license)

---

## Project Overview

The AutoML Classification System democratizes machine learning by automating complex ML workflows including:

- Data preprocessing and cleaning
- Exploratory data analysis (EDA)
- Feature engineering and transformation
- Model training with hyperparameter optimization
- Comprehensive performance evaluation
- Automated report generation

The system is built using Streamlit for the web interface, Scikit-learn for machine learning algorithms, and Plotly for interactive visualizations.

---

## Features

### Core Capabilities

| Feature | Description |
|---------|-------------|
| **Automated EDA** | Comprehensive exploratory data analysis with missing value detection, outlier analysis, and correlation studies |
| **Smart Preprocessing** | Automated data cleaning, encoding, scaling, and feature transformation based on detected issues |
| **Multi-Algorithm Training** | Support for 7 classification algorithms with automated hyperparameter tuning using Grid Search |
| **Interactive Visualizations** | Rich, interactive charts and graphs powered by Plotly for deep insights |
| **Model Comparison** | Side-by-side comparison dashboard with performance metrics and ROC curves |
| **Auto Report Generation** | Comprehensive PDF report generation summarizing the entire ML pipeline |

### Additional Features

- Drag-and-drop CSV file upload
- Real-time training progress monitoring
- Downloadable results in CSV format
- Dark-themed professional UI
- Session state management for workflow persistence

---

## Technology Stack

| Category | Technologies |
|----------|-------------|
| **Frontend** | Streamlit |
| **Backend** | Python 3.x |
| **ML Framework** | Scikit-learn |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Plotly, Matplotlib, Seaborn |
| **Imbalanced Data** | imbalanced-learn (SMOTE) |
| **PDF Generation** | FPDF / ReportLab |

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Setup Instructions

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd "ML Project"
   ```

2. **Create a virtual environment (recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**

   ```bash
   streamlit run app.py
   ```

5. **Access the application**

   Open your browser and navigate to `http://localhost:8501`

---

## Usage

### Quick Start Guide

1. **Upload Dataset**: Navigate to the Dataset Upload module and upload a CSV file
2. **Select Target**: Choose the target column for classification
3. **Review EDA**: Analyze the automated exploratory data analysis results
4. **Approve Fixes**: Review detected issues and approve preprocessing suggestions
5. **Configure Preprocessing**: Select scaling method and test set size
6. **Train Models**: Choose algorithms and start training with hyperparameter tuning
7. **Compare Results**: Evaluate model performance on the comparison dashboard
8. **Generate Report**: Download the comprehensive PDF report

### Supported File Formats

- CSV (Comma-Separated Values)

### Data Requirements

- Dataset must contain at least one target column for classification
- Numerical and categorical features are supported
- Missing values are handled automatically during preprocessing

---

## Project Structure

```
ML Project/
│
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── Project_Report.html         # HTML version of project report
├── Project_Report.pdf          # PDF version of project report
│
├── .streamlit/
│   └── config.toml             # Streamlit configuration
│
├── modules/
│   ├── __init__.py             # Module initialization
│   ├── data_upload.py          # Module 1: Dataset upload and validation
│   ├── eda.py                  # Module 2: Exploratory data analysis
│   ├── issue_detection.py      # Module 3: Issue detection and approval
│   ├── preprocessing.py        # Module 4: Data preprocessing
│   ├── model_training.py       # Module 5: Model training and tuning
│   ├── model_comparison.py     # Module 6: Model comparison dashboard
│   └── report_generation.py    # Module 7: PDF report generation
│
├── utils/
│   └── [utility functions]     # Helper functions and utilities
│
└── screenshots/
    ├── 1.png                   # Dataset Upload Interface
    ├── 2.png                   # Dataset Overview
    ├── 3.png                   # Outlier Detection
    ├── 4.png                   # EDA Summary
    ├── 5.png                   # Issue Detection
    ├── 6.png                   # Preprocessing Config
    ├── 7.png                   # Preprocessing Results
    ├── 8.png                   # Final Dataset Summary
    ├── 9.png                   # Model Training Config
    ├── 10.png                  # Model Comparison
    ├── 11.png                  # Model Rankings
    └── 12.png                  # Report Generation
```

---

## Pipeline Modules

The system follows a structured 7-module pipeline:

### Module 1: Dataset Upload and Validation

- CSV file upload with drag-and-drop support
- Automatic data type detection
- Basic statistics summary
- Target column selection

### Module 2: Automated EDA

- Missing value analysis and visualization
- Outlier detection (IQR and Z-score methods)
- Correlation matrix heatmap
- Distribution analysis for numerical features
- Categorical feature analysis

### Module 3: Issue Detection and User Approval

- Automatic identification of data quality issues
- Class imbalance detection
- Outlier flagging
- User-guided preprocessing decisions
- Checkbox-based fix approval

### Module 4: Data Preprocessing

- Outlier handling (capping/removal)
- Feature encoding (Label/One-Hot)
- Feature scaling (Standard/MinMax/Robust)
- Train-test split with stratification
- SMOTE for class imbalance

### Module 5: Model Training and Evaluation

- Algorithm selection interface
- Hyperparameter tuning (Grid Search / Random Search)
- Cross-validation support
- Real-time training progress
- Individual model metrics

### Module 6: Model Comparison Dashboard

- Side-by-side performance comparison
- Ranking by multiple metrics
- Best model identification
- CSV export of results

### Module 7: Final Report Generation

- Executive summary
- Dataset overview
- EDA visualizations
- Preprocessing documentation
- Model comparison results
- PDF download

---

## Supported Algorithms

| Algorithm | Type | Key Hyperparameters |
|-----------|------|---------------------|
| Logistic Regression | Linear | C, penalty, solver |
| K-Nearest Neighbors | Instance-based | n_neighbors, weights, metric |
| Decision Tree | Tree-based | max_depth, min_samples_split, criterion |
| Random Forest | Ensemble | n_estimators, max_depth, min_samples_leaf |
| Naive Bayes (Gaussian) | Probabilistic | var_smoothing |
| Support Vector Machine | Kernel-based | C, kernel, gamma |
| OneR (Rule-Based) | Rule-based | n_bins |

---

## Evaluation Metrics

The system evaluates model performance using the following metrics:

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall proportion of correct predictions |
| **Precision** | Ratio of true positives to predicted positives |
| **Recall** | Ratio of true positives to actual positives |
| **F1-Score** | Harmonic mean of precision and recall |
| **ROC-AUC** | Area under the Receiver Operating Characteristic curve |
| **Confusion Matrix** | Detailed breakdown of prediction outcomes |
| **Training Time** | Time taken to train each model |

---

## Screenshots

### Dataset Upload and Overview

- **Figure 1**: Dataset Upload Interface
- **Figure 2**: Dataset Overview with Statistics

### Exploratory Data Analysis

- **Figure 3**: Outlier Detection using IQR Method
- **Figure 4**: Distribution Statistics and EDA Summary

### Issue Detection and Preprocessing

- **Figure 5**: Issue Detection Dashboard
- **Figure 6**: Preprocessing Configuration
- **Figure 7**: Preprocessing Results Log
- **Figure 8**: Final Dataset Summary

### Model Training and Comparison

- **Figure 9**: Model Training Configuration
- **Figure 10**: Model Comparison Metrics
- **Figure 11**: Model Rankings Dashboard

### Report Generation

- **Figure 12**: Final Report Generation Interface

All screenshots are available in the `/screenshots` directory.

---

## Contributors

| Name | Roll Number |
|------|-------------|
| Muhammad Ahmad | 467360 |
| Muhammad Tayyab Rizwan | 463804 |
| Mudassir Ahmed Sheikh | 454473 |

**Instructor**: Dr Syed Imran Malik

---

## License

This project is developed for educational purposes as part of the Machine Learning course.

---

## Acknowledgments

- Streamlit for the web application framework
- Scikit-learn for machine learning algorithms
- Plotly for interactive visualizations
- The open-source Python community

---

## Support

For questions or issues, please contact the project contributors or refer to the project documentation.

---

**AutoML Classification System** | Machine Learning Project | December 2024