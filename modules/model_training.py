"""
Module 5: Model Training & Hyperparameter Optimization

This module trains and evaluates 7 classification algorithms with automated
hyperparameter tuning and comprehensive performance metrics.
"""

import pandas as pd
import numpy as np
import streamlit as st
import time
from typing import Dict, List, Tuple, Any
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)


# ==================== HYPERPARAMETER GRIDS ====================

PARAM_GRIDS = {
    'logistic_regression': {
        'C': [0.01, 0.1, 1, 10],
        'penalty': ['l2'],
        'solver': ['lbfgs', 'liblinear'],
        'max_iter': [1000, 2000]  # Increased to prevent convergence warnings
    },
    
    'knn': {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    },
    
    'decision_tree': {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 5, 10, 15, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    
    'naive_bayes': {
        'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
    },
    
    'random_forest': {
        'n_estimators': [50, 100, 200],
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    },
    
    'svm': {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }
}


# ==================== RULE-BASED CLASSIFIER (OneR) ====================

class RuleBasedClassifier:
    """
    OneR (One Rule) Classifier - Uses a single best rule for classification.
    
    OneR finds the feature that best discriminates between classes by:
    1. For each feature, find the best split threshold
    2. Calculate error rate for that feature
    3. Select the feature with the lowest error rate
    
    Reference: Holte, R. C. (1993). Very simple classification rules perform 
    well on most commonly used datasets.
    """
    
    def __init__(self, n_bins=10):
        """
        Args:
            n_bins: Number of bins for discretizing continuous features
        """
        self.n_bins = n_bins
        self.best_feature = None
        self.best_feature_name = None
        self.best_threshold = None
        self.best_rule = None
        self.classes = None
        self.feature_names = None
        self.error_rate = None
    
    def fit(self, X, y):
        """
        Learn the single best rule from training data.
        
        For each feature:
        1. Try different thresholds
        2. Calculate error for each threshold
        3. Keep feature with minimum error
        """
        self.classes = np.unique(y)
        self.feature_names = X.columns.tolist() if hasattr(X, 'columns') else [f'feature_{i}' for i in range(X.shape[1])]
        
        # Convert to numpy if DataFrame
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        best_error = float('inf')
        
        # Try each feature
        for feat_idx in range(X_array.shape[1]):
            feature_values = X_array[:, feat_idx]
            
            # Try different thresholds for this feature
            # Use percentiles as candidate thresholds
            unique_values = np.unique(feature_values)
            
            if len(unique_values) > self.n_bins:
                # Use quantiles for continuous features
                thresholds = np.percentile(feature_values, np.linspace(10, 90, min(9, len(unique_values)-1)))
            else:
                # Use actual values for discrete features
                thresholds = unique_values[:-1]  # All except last
            
            # Try each threshold
            for threshold in thresholds:
                # Make predictions with this threshold
                predictions = np.where(feature_values <= threshold, self.classes[0], self.classes[1])
                
                # Calculate error
                error = np.mean(predictions != y)
                
                # Keep if best so far
                if error < best_error:
                    best_error = error
                    self.best_feature = feat_idx
                    self.best_feature_name = self.feature_names[feat_idx]
                    self.best_threshold = threshold
                    self.error_rate = error
                    # Determine which class for each side
                    left_class = np.bincount(y[feature_values <= threshold]).argmax()
                    right_class = np.bincount(y[feature_values > threshold]).argmax()
                    self.best_rule = {
                        'feature_idx': feat_idx,
                        'feature_name': self.best_feature_name,
                        'threshold': threshold,
                        'class_if_less_equal': left_class,
                        'class_if_greater': right_class
                    }
        
        return self
    
    def predict(self, X):
        """
        Apply the single best rule to make predictions.
        """
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        feature_values = X_array[:, self.best_feature]
        
        # Apply the ONE rule
        predictions = np.where(
            feature_values <= self.best_threshold,
            self.best_rule['class_if_less_equal'],
            self.best_rule['class_if_greater']
        )
        
        return predictions
    
    def get_rules(self):
        """
        Return the single rule in human-readable format.
        """
        if self.best_rule is None:
            return ["No rule learned yet. Please fit the model first."]
        
        rule_str = (
            f"OneR Rule: IF {self.best_feature_name}  {self.best_threshold:.3f} "
            f"THEN Class {self.best_rule['class_if_less_equal']} "
            f"ELSE Class {self.best_rule['class_if_greater']}\n"
            f"(Training Error Rate: {self.error_rate*100:.2f}%)"
        )
        
        return [rule_str]


# ==================== TRAINING FUNCTIONS ====================

def train_model_with_tuning(
    model_name: str,
    base_model: Any,
    param_grid: Dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    use_grid_search: bool = True,
    cv_folds: int = 3  # Reduced for speed
) -> Tuple[Any, Dict, float]:
    """
    Train and tune a single model.
    
    Args:
        model_name: Name of the model
        base_model: Untrained sklearn model
        param_grid: Hyperparameter grid
        X_train: Training features
        y_train: Training labels
        use_grid_search: Use GridSearch vs RandomizedSearch
        cv_folds: Cross-validation folds
        
    Returns:
        Best model, best params, training time
    """
    start_time = time.time()
    
    try:
        if use_grid_search:
            search = GridSearchCV(
                base_model,
                param_grid,
                cv=cv_folds,
                scoring='f1_weighted',
                n_jobs=-1,  # Use all cores
                verbose=0
            )
        else:
            search = RandomizedSearchCV(
                base_model,
                param_grid,
                n_iter=20,
                cv=cv_folds,
                scoring='f1_weighted',
                n_jobs=-1,
                random_state=42,
                verbose=0
            )
        
        search.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        return search.best_estimator_, search.best_params_, training_time
    
    except Exception as e:
        st.warning(f" Error tuning {model_name}: {str(e)}. Using default parameters.")
        # Fallback to default model
        start_time = time.time()
        base_model.fit(X_train, y_train)
        training_time = time.time() - start_time
        return base_model, {}, training_time


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    class_names: List = None
) -> Dict:
    """
    Evaluate model and calculate all metrics.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        model_name: Name of the model
        class_names: List of class names
        
    Returns:
        Dictionary with all metrics
    """
    # Make predictions
    start_time = time.time()
    y_pred = model.predict(X_test)
    prediction_time = time.time() - start_time
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    # ROC-AUC for binary classification
    roc_auc = None
    if len(np.unique(y_test)) == 2:
        try:
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)[:, 1]
                roc_auc = roc_auc_score(y_test, y_proba)
            elif hasattr(model, 'decision_function'):
                y_score = model.decision_function(X_test)
                roc_auc = roc_auc_score(y_test, y_score)
        except:
            roc_auc = None
    
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'classification_report': report,
        'roc_auc': roc_auc,
        'prediction_time': prediction_time
    }


def train_all_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    use_grid_search: bool = True,
    selected_models: List[str] = None
) -> Dict:
    """
    Train all selected models with hyperparameter tuning.
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        use_grid_search: Use GridSearch vs RandomizedSearch
        selected_models: List of model names to train (None = all)
        
    Returns:
        Dictionary with all results
    """
    if selected_models is None:
        selected_models = [
            'logistic_regression', 'knn', 'decision_tree',
            'naive_bayes', 'random_forest', 'svm', 'rule_based'
        ]
    
    results = {}
    
    # Define base models
    base_models = {
        'logistic_regression': ('Logistic Regression', LogisticRegression(random_state=42)),
        'knn': ('K-Nearest Neighbors', KNeighborsClassifier()),
        'decision_tree': ('Decision Tree', DecisionTreeClassifier(random_state=42)),
        'naive_bayes': ('Naive Bayes', GaussianNB()),
        'random_forest': ('Random Forest', RandomForestClassifier(random_state=42)),
        'svm': ('Support Vector Machine', SVC(random_state=42)),
        'rule_based': ('Rule-Based Classifier', RuleBasedClassifier())
    }
    
    total_models = len(selected_models)
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, model_key in enumerate(selected_models):
        if model_key not in base_models:
            continue
        
        display_name, base_model = base_models[model_key]
        status_text.text(f"Training {display_name}... ({idx + 1}/{total_models})")
        
        # Train model
        if model_key == 'rule_based':
            # Rule-based doesn't use hyperparameter tuning
            start_time = time.time()
            trained_model = base_model.fit(X_train, y_train)
            training_time = time.time() - start_time
            best_params = {}
        else:
            param_grid = PARAM_GRIDS.get(model_key, {})
            trained_model, best_params, training_time = train_model_with_tuning(
                display_name,
                base_model,
                param_grid,
                X_train,
                y_train,
                use_grid_search=use_grid_search
            )
        
        # Evaluate model
        metrics = evaluate_model(trained_model, X_test, y_test, display_name)
        
        # Store results
        results[model_key] = {
            'model': trained_model,
            'display_name': display_name,
            'best_params': best_params,
            'training_time': training_time,
            'metrics': metrics
        }
        
        # Update progress
        progress_bar.progress((idx + 1) / total_models)
    
    progress_bar.empty()
    status_text.empty()
    
    return results


# ==================== UI FUNCTIONS ====================

def run_module_5() -> None:
    """
    Main function for Module 5: Model Training & Evaluation.
    """
    st.markdown("""
        <div style="text-align: center; padding: 1.5rem 0;">
            <h1 style="font-size: 3rem; margin-bottom: 0.5rem;"> Model Training & Evaluation</h1>
            <p style="font-size: 1.15rem; color: #6b7280; font-weight: 400;">
                Train and compare classification algorithms
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Check dependencies
    preprocessed_data = st.session_state.get('preprocessed_data')
    skip_preprocessing = st.session_state.get('skip_preprocessing', False)
    
    if not skip_preprocessing and not preprocessed_data:
        st.error(" No preprocessed data found. Please complete preprocessing first.")
        if st.button(" Go to Preprocessing"):
            st.session_state.current_step = 4
            st.rerun()
        return
    
    # Get data and class labels
    if skip_preprocessing:
        # Use raw data with basic preparation
        df = st.session_state.df
        target_column = st.session_state.target_column
        
        # Store original class labels before encoding
        original_classes = df[target_column].unique()
        
        # Encode target if needed
        from sklearn.preprocessing import LabelEncoder
        if not pd.api.types.is_numeric_dtype(df[target_column]):
            le = LabelEncoder()
            df[target_column] = le.fit_transform(df[target_column])
            class_labels = le.classes_.tolist()
        else:
            class_labels = sorted(original_classes.tolist())
        
        # Simple train/test split
        from sklearn.model_selection import train_test_split
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Handle categorical columns (simple one-hot)
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        st.info(" **Note**: Using raw data with basic preparation (preprocessing was skipped)")
    else:
        X_train = preprocessed_data['X_train']
        X_test = preprocessed_data['X_test']
        y_train = preprocessed_data['y_train']
        y_test = preprocessed_data['y_test']
        
        # Get class labels from encoder if available
        encoders = preprocessed_data.get('encoders', {})
        if 'target' in encoders and 'classes' in encoders['target']:
            class_labels = encoders['target']['classes']
        else:
            # Fallback to numeric labels
            class_labels = sorted(np.unique(y_train).tolist())
    
    # Store class labels in session state for confusion matrix display
    st.session_state.class_labels = class_labels
    
    # Check if already completed
    training_completed = st.session_state.get('training_completed', False)
    
    if training_completed:
        st.info(" **Training already completed!** Displaying previous results. Click 'Re-train' to train again.")
        
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button(" Re-train Models", use_container_width=True):
                st.session_state.training_completed = False
                st.session_state.trained_models = None
                st.session_state.training_results = None
                st.rerun()
    
    # Dataset summary
    st.markdown("##  Dataset Summary")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Training Samples", f"{len(X_train):,}")
    with col2:
        st.metric("Test Samples", f"{len(X_test):,}")
    with col3:
        st.metric("Features", X_train.shape[1])
    with col4:
        n_classes = len(np.unique(y_train))
        st.metric("Classes", n_classes)
    
    st.markdown("---")
    
    # Configuration
    if not training_completed:
        st.markdown("##  Training Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            use_grid_search = st.radio(
                "**Hyperparameter Tuning Strategy:**",
                options=[True, False],
                format_func=lambda x: "Grid Search (Exhaustive)" if x else "Randomized Search (Faster)",
                help="Grid Search is more thorough but slower"
            )
        
        with col2:
            st.markdown("**Models to Train:**")
            all_selected = st.checkbox("Select All", value=True, key="select_all_models")
            
            if all_selected:
                selected_models = [
                    'logistic_regression', 'knn', 'decision_tree',
                    'naive_bayes', 'random_forest', 'svm', 'rule_based'
                ]
            else:
                selected_models = []
                if st.checkbox("Logistic Regression", value=True):
                    selected_models.append('logistic_regression')
                if st.checkbox("K-Nearest Neighbors", value=True):
                    selected_models.append('knn')
                if st.checkbox("Decision Tree", value=True):
                    selected_models.append('decision_tree')
                if st.checkbox("Naive Bayes", value=True):
                    selected_models.append('naive_bayes')
                if st.checkbox("Random Forest", value=True):
                    selected_models.append('random_forest')
                if st.checkbox("SVM", value=True):
                    selected_models.append('svm')
                if st.checkbox("Rule-Based", value=True):
                    selected_models.append('rule_based')
        
        st.markdown("---")
        
        # Train button
        if st.button(" Train All Models", type="primary", use_container_width=True):
            if not selected_models:
                st.error(" Please select at least one model to train.")
            else:
                st.markdown("##  Training Progress")
                
                with st.spinner("Training models..."):
                    try:
                        results = train_all_models(
                            X_train, X_test, y_train, y_test,
                            use_grid_search=use_grid_search,
                            selected_models=selected_models
                        )
                        
                        # Store results
                        st.session_state.trained_models = results
                        st.session_state.training_completed = True
                        
                        st.success(" **Training Complete!**")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f" **Training Error**: {str(e)}")
                        st.exception(e)
    
    # Display results if completed
    if training_completed and st.session_state.get('trained_models'):
        st.markdown("##  Model Comparison")
        
        results = st.session_state.trained_models
        
        # Create comparison table
        comparison_data = []
        for model_key, result in results.items():
            metrics = result['metrics']
            comparison_data.append({
                'Model': result['display_name'],
                'Accuracy': f"{metrics['accuracy'] * 100:.2f}%",
                'Precision': f"{metrics['precision']:.3f}",
                'Recall': f"{metrics['recall']:.3f}",
                'F1-Score': f"{metrics['f1_score']:.3f}",
                'ROC-AUC': f"{metrics['roc_auc']:.3f}" if metrics['roc_auc'] else 'N/A',
                'Training Time': f"{result['training_time']:.2f}s",
                '_sort_f1': metrics['f1_score'],
                '_sort_accuracy': metrics['accuracy'],
                '_sort_time': result['training_time']
            })
        
        # Sort by F1-score (primary), Accuracy (secondary), Training Time (tertiary - ascending)
        comparison_data.sort(key=lambda x: (-x['_sort_f1'], -x['_sort_accuracy'], x['_sort_time']))
        
        # Remove sort keys
        for row in comparison_data:
            del row['_sort_f1']
            del row['_sort_accuracy']
            del row['_sort_time']
        
        # Display table
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        # ROC-AUC explanation
        st.caption(" **Note**: ROC-AUC is only calculated for binary classification problems. Shows 'N/A' for multi-class problems.")
        
        # Highlight best model
        best_model_name = comparison_data[0]['Model']
        st.success(f" **Best Model**: {best_model_name} (Ranked by F1-Score, then Accuracy, then Speed)")
        
        st.markdown("---")
        
        # Detailed metrics for each model
        st.markdown("##  Detailed Metrics")
        
        for model_key, result in results.items():
            with st.expander(f"**{result['display_name']}** - Detailed Metrics"):
                metrics = result['metrics']
                
                # Best parameters
                if result['best_params']:
                    st.markdown("**Best Hyperparameters:**")
                    st.json(result['best_params'])
                
                # Metrics
                st.markdown("**Performance Metrics:**")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
                with col2:
                    st.metric("Precision", f"{metrics['precision']:.3f}")
                with col3:
                    st.metric("Recall", f"{metrics['recall']:.3f}")
                with col4:
                    st.metric("F1-Score", f"{metrics['f1_score']:.3f}")
                
                # Confusion Matrix with actual class labels
                st.markdown("**Confusion Matrix:**")
                cm = metrics['confusion_matrix']
                
                # Get actual class labels
                class_labels = st.session_state.get('class_labels', [f'Class {i}' for i in range(len(cm))])
                
                # Ensure we have the right number of labels
                if len(class_labels) != len(cm):
                    class_labels = [f'Class {i}' for i in range(len(cm))]
                
                # Create readable label mapping for display
                label_map = {i: str(label) for i, label in enumerate(class_labels)}
                
                # Create DataFrame with actual class names
                cm_df = pd.DataFrame(
                    cm,
                    index=[f'Actually "{label_map[i]}"' for i in range(len(cm))],
                    columns=[f'Predicted "{label_map[i]}"' for i in range(len(cm))]
                )
                
                # Display confusion matrix
                st.dataframe(cm_df, use_container_width=True)
                
                # Add detailed explanation with class meanings
                st.caption(" **How to read this matrix:**")
                st.caption(
                    f" **Rows** show the ACTUAL class (ground truth)\n"
                    f" **Columns** show what the model PREDICTED\n"
                    f" **Diagonal values** ( correct predictions): "
                    f"{', '.join([f'{cm[i][i]} {label_map[i]}{label_map[i]}' for i in range(len(cm))])}\n"
                    f" **Off-diagonal values** ( mistakes): Where the model got it wrong"
                )
                
                # Show specific misclassifications if any
                misclassifications = []
                for i in range(len(cm)):
                    for j in range(len(cm)):
                        if i != j and cm[i][j] > 0:
                            misclassifications.append(
                                f"{cm[i][j]} times: Actually '{label_map[i]}' but predicted '{label_map[j]}'"
                            )
                
                if misclassifications:
                    st.markdown("** Misclassifications:**")
                    for misc in misclassifications:
                        st.write(f" {misc}")

                
                # Training time
                st.caption(f" Training Time: {result['training_time']:.2f}s | Prediction Time: {metrics['prediction_time']:.4f}s")
                
                # Rules for rule-based classifier
                if model_key == 'rule_based' and hasattr(result['model'], 'get_rules'):
                    st.markdown("**Learned Rules:**")
                    rules = result['model'].get_rules()
                    for rule in rules:
                        st.code(rule)
        
        st.markdown("---")
        
        st.success("""
             **All models trained successfully!**
            
            You can now proceed to compare models in detail and generate the final report.
        """)
    
    # Navigation
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        back_step = 4 if not skip_preprocessing else 3
        back_label = " Back to Preprocessing" if not skip_preprocessing else " Back to Issues"
        if st.button(back_label, use_container_width=True):
            st.session_state.current_step = back_step
            st.rerun()
    with col3:
        if training_completed:
            if st.button("Continue to Comparison ", use_container_width=True, type="primary"):
                st.session_state.current_step = 6
                st.rerun()
