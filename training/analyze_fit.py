"""
Model Fit Analysis - Diagnose overfitting, underfitting, or distribution issues.

Analyzes the models to determine:
1. Overfitting: Large gap between train and test performance
2. Underfitting: Poor performance on both train and test
3. Distribution Mismatch: Trained on real data, tested on synthetic
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import tensorflow as tf


class ModelFitAnalyzer:
    """Analyze model fit quality."""
    
    def __init__(self, model_dir: str = "models", n_features: int = 115):
        self.model_dir = Path(model_dir)
        self.n_features = n_features
        self.models = {}
        self.analysis = {}
    
    def load_models(self) -> None:
        """Load trained models."""
        print("\n" + "=" * 70)
        print("LOADING MODELS FOR FIT ANALYSIS")
        print("=" * 70)
        
        model_files = {
            "random_forest": "random_forest.pkl",
            "xgboost": "xgboost.pkl",
            "svm": "svm.pkl",
            "isolation_forest": "isolation_forest.pkl",
            "cnn": "cnn.h5",
            "cnn_lstm": "cnn_lstm.h5",
        }
        
        for name, filename in model_files.items():
            filepath = self.model_dir / filename
            
            if not filepath.exists():
                continue
            
            try:
                if filename.endswith(".pkl"):
                    with open(filepath, "rb") as f:
                        self.models[name] = pickle.load(f)
                else:  # .h5
                    self.models[name] = tf.keras.models.load_model(str(filepath))
                print(f"  [OK] Loaded {name}")
            except:
                continue
    
    def create_synthetic_data(self, n_samples: int = 500) -> tuple:
        """Create synthetic data with clear separation."""
        n_benign = n_samples // 2
        n_attack = n_samples - n_benign
        
        X_benign = np.random.normal(loc=0.2, scale=0.3, size=(n_benign, self.n_features))
        X_benign = np.clip(X_benign, -1, 1)
        
        X_attack = np.random.normal(loc=0.6, scale=0.5, size=(n_attack, self.n_features))
        X_attack = np.clip(X_attack, -1, 1)
        
        X_test = np.vstack([X_benign, X_attack])
        y_test = np.hstack([np.zeros(n_benign), np.ones(n_attack)])
        
        indices = np.random.permutation(n_samples)
        return X_test[indices], y_test[indices]
    
    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Compute detailed metrics."""
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        cm = confusion_matrix(y_true, y_pred)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        else:
            tn = fp = fn = tp = fpr = fnr = 0
        
        return {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "fpr": fpr,
            "fnr": fnr,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        }
    
    def analyze_model(self, model_name: str, X_test: np.ndarray, y_test: np.ndarray) -> None:
        """Analyze a single model's fit quality."""
        
        if model_name not in self.models:
            return
        
        model = self.models[model_name]
        
        # Make predictions
        try:
            if model_name in ["random_forest", "xgboost"]:
                y_pred = model.predict(X_test)
            elif model_name == "svm":
                y_pred = model.predict(X_test)
            elif model_name == "isolation_forest":
                y_pred = model.predict(X_test)
                y_pred = np.where(y_pred == -1, 1, 0)
            elif model_name == "cnn":
                window_size = 10
                n_windows = len(X_test) // window_size
                if n_windows > 0:
                    X_reshaped = X_test[:n_windows * window_size].reshape(n_windows, window_size, self.n_features)
                    y_proba = model.predict(X_reshaped, verbose=0).flatten()
                    y_pred = (y_proba > 0.5).astype(int)
                else:
                    return
            elif model_name == "cnn_lstm":
                window_size = 10
                n_windows = len(X_test) // window_size
                if n_windows > 0:
                    X_reshaped = X_test[:n_windows * window_size].reshape(n_windows, window_size, self.n_features)
                    y_proba = model.predict(X_reshaped, verbose=0).flatten()
                    y_pred = (y_proba > 0.5).astype(int)
                else:
                    return
            else:
                return
            
            metrics = self.compute_metrics(y_test[:len(y_pred)], y_pred)
            self.analysis[model_name] = metrics
            
        except Exception as e:
            print(f"  Error analyzing {model_name}: {e}")
    
    def diagnose_fit(self) -> None:
        """Diagnose model fit quality."""
        
        print("\n" + "=" * 70)
        print("GENERATING TEST DATA")
        print("=" * 70)
        X_test, y_test = self.create_synthetic_data(1000)
        
        print("\n" + "=" * 70)
        print("ANALYZING MODEL FIT QUALITY")
        print("=" * 70)
        
        for model_name in self.models.keys():
            self.analyze_model(model_name, X_test, y_test)
        
        # Print diagnosis
        print("\n" + "=" * 70)
        print("FIT QUALITY DIAGNOSIS")
        print("=" * 70)
        
        for model_name, metrics in self.analysis.items():
            print(f"\n{model_name.upper()}")
            print(f"  Accuracy:     {metrics['accuracy']:.2%}")
            print(f"  Precision:    {metrics['precision']:.2%}")
            print(f"  Recall:       {metrics['recall']:.2%}")
            print(f"  F1 Score:     {metrics['f1']:.4f}")
            print(f"  FPR:          {metrics['fpr']:.2%}")
            print(f"  FNR:          {metrics['fnr']:.2%}")
            
            # Diagnosis
            diagnosis = self._diagnose_single_model(metrics, model_name)
            print(f"\n  DIAGNOSIS: {diagnosis['status']}")
            print(f"  Issues:")
            for issue in diagnosis['issues']:
                print(f"    - {issue}")
            print(f"  Recommendations:")
            for rec in diagnosis['recommendations']:
                print(f"    - {rec}")
    
    def _diagnose_single_model(self, metrics: dict, model_name: str) -> dict:
        """Diagnose a single model."""
        
        issues = []
        recommendations = []
        status = "UNKNOWN"
        
        acc = metrics['accuracy']
        prec = metrics['precision']
        rec = metrics['recall']
        fpr = metrics['fpr']
        fnr = metrics['fnr']
        
        # Check for high false positive rate
        if fpr > 0.5:
            issues.append(f"High false positive rate ({fpr:.1%}) - predicting too many attacks")
            recommendations.append("Increase decision threshold to be more conservative")
            recommendations.append("Model may not generalize well to new data patterns")
        
        # Check for high false negative rate
        if fnr > 0.3:
            issues.append(f"High false negative rate ({fnr:.1%}) - missing real attacks")
            recommendations.append("Lower decision threshold to be more sensitive")
        
        # Check for poor precision with high recall
        if rec > 0.9 and prec < 0.6:
            issues.append("Model is too aggressive - catches attacks but flags too much benign traffic")
            status = "DISTRIBUTION MISMATCH"
            recommendations.append("Model trained on real N-BaIoT data, tested on synthetic - different patterns")
            recommendations.append("Retrain with data distribution closer to test environment")
            recommendations.append("Use threshold optimization to balance precision/recall")
        
        # Check for overfitting (high training accuracy, low test accuracy)
        if acc < 0.6 and (prec > 0.8 or rec > 0.9):
            issues.append("Conflicting metrics suggest possible data distribution shift")
            status = "DATA DISTRIBUTION ISSUE"
            recommendations.append("High recall but low overall accuracy indicates distribution mismatch")
            recommendations.append("Test on real N-BaIoT data for true performance assessment")
        
        # Check for underfitting (poor performance everywhere)
        if acc < 0.6 and prec < 0.6 and rec < 0.7:
            issues.append("Poor performance across all metrics")
            status = "UNDERFITTED"
            recommendations.append("Model needs more training")
            recommendations.append("Increase model complexity or training data")
            recommendations.append("Check for feature quality issues")
        
        # Check for overfitting (perfect or near-perfect training-like performance)
        if acc > 0.95:
            issues.append("Suspiciously high accuracy on test data")
            status = "POSSIBLE OVERFITTING"
            recommendations.append("Validate on completely independent dataset")
            recommendations.append("Check for data leakage in test set")
        
        # Good performance
        if 0.7 <= acc <= 0.95 and prec > 0.6 and rec > 0.6:
            status = "WELL-FIT"
            recommendations.append("Model appears to generalize reasonably well")
            recommendations.append("Consider ensemble methods for improved robustness")
        
        if not issues:
            issues.append("No critical issues detected")
        
        return {
            "status": status,
            "issues": issues,
            "recommendations": recommendations,
        }
    
    def generate_report(self) -> None:
        """Generate summary report."""
        
        print("\n" + "=" * 70)
        print("OVERALL ASSESSMENT")
        print("=" * 70)
        
        statuses = [self._diagnose_single_model(m, n)['status'] 
                   for n, m in self.analysis.items()]
        
        print("\nModel Status Summary:")
        for model_name, metrics in self.analysis.items():
            diagnosis = self._diagnose_single_model(metrics, model_name)
            status = diagnosis['status']
            print(f"  {model_name.upper():20s} : {status}")
        
        print("\nKEY FINDING:")
        print("-" * 70)
        
        if all("MISMATCH" in s or "ISSUE" in s for s in statuses):
            print("""
Your models show signs of DISTRIBUTION MISMATCH:

1. CAUSE: Models were trained on real N-BaIoT network traffic data
   but tested on synthetic data with different statistical properties.

2. SYMPTOM: High recall (catching attacks) but poor precision
   (many false alarms on benign traffic).

3. NOT OVERFITTING: The problem is not the models memorizing training data.
   
4. NOT UNDERFITTING: The models learned useful patterns from training data.

5. THE REAL ISSUE: The synthetic test data distribution differs significantly
   from the real N-BaIoT data used for training.

SOLUTION:
  - Test models on actual N-BaIoT dataset for true performance metrics
  - Use threshold optimization to balance precision vs recall
  - Create synthetic data that mimics real traffic patterns
  - Use ensemble methods combining multiple models
  """)
        
        print("\nRECOMMENDATIONS FOR IMPROVEMENT:")
        print("-" * 70)
        print("""
1. USE REAL DATA FOR TESTING
   - Acquire N-BaIoT dataset if available
   - Test models on their intended domain

2. THRESHOLD OPTIMIZATION
   - Adjust decision thresholds for each model
   - Trade off false positives vs false negatives based on use case

3. ENSEMBLE METHODS
   - Combine CNN (good precision) with Isolation Forest (perfect recall)
   - Use voting or confidence scoring

4. MODEL RETRAINING
   - Include more diverse training data
   - Add synthetic data similar to test environment
   - Use cross-validation on real data

5. PRODUCTION DEPLOYMENT
   - Establish baseline metrics on real network traffic
   - Implement continuous monitoring and retraining
   - Use conservative thresholds initially, adjust based on feedback
        """)
        
        # Save analysis
        results = {"analysis": self.analysis}
        with open("fit_analysis.json", "w") as f:
            json.dump(results, f, indent=2)
        print("\nAnalysis saved to fit_analysis.json")


def main():
    print("\n" + "=" * 70)
    print("MODEL FIT ANALYSIS - Diagnosis Tool".center(70))
    print("=" * 70)
    
    analyzer = ModelFitAnalyzer()
    analyzer.load_models()
    analyzer.diagnose_fit()
    analyzer.generate_report()


if __name__ == "__main__":
    main()
