"""
Optimized model testing with focus on reducing false positive rates.

This script tests models with threshold optimization to minimize false positives
while maintaining reasonable detection rates.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import tensorflow as tf


class OptimizedModelTester:
    """Test trained models with FPR optimization."""
    
    def __init__(self, model_dir: str = "models", n_features: int = 115):
        self.model_dir = Path(model_dir)
        self.n_features = n_features
        self.models = {}
        self.metrics = {}
        self.optimized_thresholds = {}
    
    def load_models(self) -> None:
        """Load trained models."""
        print("\n" + "=" * 70)
        print("LOADING MODELS")
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
                print(f"  [MISSING] {name}: {filename} not found")
                continue
            
            try:
                if filename.endswith(".pkl"):
                    with open(filepath, "rb") as f:
                        self.models[name] = pickle.load(f)
                else:  # .h5
                    self.models[name] = tf.keras.models.load_model(str(filepath))
                
                print(f"  [OK] Loaded {name}")
            except Exception as e:
                print(f"  [ERROR] Failed to load {name}: {type(e).__name__}")
    
    def create_synthetic_data(self, n_samples: int = 500, n_benign: int = None) -> tuple:
        """Create synthetic test data."""
        if n_benign is None:
            n_benign = n_samples // 2
        n_attack = n_samples - n_benign
        
        # Generate benign samples
        X_benign = np.random.normal(loc=0.2, scale=0.3, size=(n_benign, self.n_features))
        X_benign = np.clip(X_benign, -1, 1)
        
        # Generate attack samples
        X_attack = np.random.normal(loc=0.6, scale=0.5, size=(n_attack, self.n_features))
        X_attack = np.clip(X_attack, -1, 1)
        
        # Combine and create labels
        X_test = np.vstack([X_benign, X_attack])
        y_test = np.hstack([np.zeros(n_benign), np.ones(n_attack)])
        
        # Shuffle
        indices = np.random.permutation(n_samples)
        X_test = X_test[indices]
        y_test = y_test[indices]
        
        print(f"\n  Created {n_samples} synthetic samples:")
        print(f"    - Benign: {n_benign}")
        print(f"    - Attack: {n_attack}")
        
        return X_test, y_test
    
    def get_probabilities(self, model_name: str, X_test: np.ndarray, y_test: np.ndarray) -> np.ndarray:
        """Get probability predictions for threshold optimization."""
        model = self.models[model_name]
        
        if model_name == "random_forest":
            return model.predict_proba(X_test)[:, 1]
        elif model_name == "xgboost":
            return model.predict_proba(X_test)[:, 1]
        elif model_name == "svm":
            # SVM doesn't have predict_proba, use decision function
            decision = model.decision_function(X_test)
            # Normalize to 0-1 range
            return 1 / (1 + np.exp(-decision))
        elif model_name == "isolation_forest":
            # Isolation forest uses anomaly scores
            scores = model.score_samples(X_test)
            # Convert to probabilities (inverted, as more negative = anomaly)
            return 1 / (1 + np.exp(scores))
        elif model_name == "cnn":
            window_size = 10
            n_full_windows = len(X_test) // window_size
            if n_full_windows > 0:
                X_reshaped = X_test[:n_full_windows * window_size].reshape(n_full_windows, window_size, self.n_features)
                proba = model.predict(X_reshaped, verbose=0).flatten()
                # Pad to match original length
                return np.pad(proba, (0, len(X_test) - len(proba)), mode='edge')
            return np.zeros(len(X_test))
        elif model_name == "cnn_lstm":
            window_size = 10
            n_full_windows = len(X_test) // window_size
            if n_full_windows > 0:
                X_reshaped = X_test[:n_full_windows * window_size].reshape(n_full_windows, window_size, self.n_features)
                proba = model.predict(X_reshaped, verbose=0).flatten()
                return np.pad(proba, (0, len(X_test) - len(proba)), mode='edge')
            return np.zeros(len(X_test))
        
        return np.zeros(len(X_test))
    
    def find_optimal_threshold(self, y_true: np.ndarray, y_proba: np.ndarray, target_fpr: float = 0.1) -> float:
        """
        Find optimal threshold to achieve target false positive rate.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            target_fpr: Target false positive rate (default 10%)
        
        Returns:
            Optimal threshold
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        
        # Find threshold closest to target FPR
        idx = np.argmin(np.abs(fpr - target_fpr))
        optimal_threshold = thresholds[idx]
        
        return optimal_threshold
    
    def test_with_threshold(self, model_name: str, X_test: np.ndarray, y_test: np.ndarray, 
                           target_fpr: float = 0.1) -> dict:
        """Test model with optimized threshold."""
        
        # Get probabilities
        y_proba = self.get_probabilities(model_name, X_test, y_test)
        
        if np.all(y_proba == 0):
            return None
        
        # Find optimal threshold
        threshold = self.find_optimal_threshold(y_test, y_proba, target_fpr)
        self.optimized_thresholds[model_name] = float(threshold)
        
        # Make predictions with threshold
        y_pred = (y_proba >= threshold).astype(int)
        
        # Compute metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        cm = confusion_matrix(y_test, y_pred)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        else:
            fpr = fnr = 0
        
        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "false_positive_rate": float(fpr),
            "false_negative_rate": float(fnr),
            "threshold": float(threshold),
            "tp": int(tp) if cm.size == 4 else 0,
            "tn": int(tn) if cm.size == 4 else 0,
            "fp": int(fp) if cm.size == 4 else 0,
            "fn": int(fn) if cm.size == 4 else 0,
        }
    
    def run_optimized_tests(self, n_samples: int = 1000, target_fpr: float = 0.10) -> None:
        """Run tests with FPR optimization."""
        
        # Create synthetic data
        print("\n" + "=" * 70)
        print("GENERATING SYNTHETIC TEST DATA")
        print("=" * 70)
        X_test, y_test = self.create_synthetic_data(n_samples)
        
        # Test models
        print("\n" + "=" * 70)
        print(f"TESTING MODELS (Target FPR: {target_fpr*100:.1f}%)")
        print("=" * 70)
        
        for model_name in self.models.keys():
            try:
                result = self.test_with_threshold(model_name, X_test, y_test, target_fpr)
                
                if result is None:
                    print(f"\n  {model_name.upper()}: SKIPPED (no probabilities)")
                    continue
                
                self.metrics[model_name] = result
                
                print(f"\n  {model_name.upper()}")
                print(f"    Accuracy:  {result['accuracy']:.4f}")
                print(f"    Precision: {result['precision']:.4f}")
                print(f"    Recall:    {result['recall']:.4f}")
                print(f"    F1 Score:  {result['f1']:.4f}")
                print(f"    FPR:       {result['false_positive_rate']:.4f} (target: {target_fpr:.4f})")
                print(f"    Threshold: {result['threshold']:.4f}")
                print(f"    Confusion: TP={result['tp']}, TN={result['tn']}, FP={result['fp']}, FN={result['fn']}")
                
            except Exception as e:
                print(f"\n  {model_name.upper()}: ERROR - {str(e)[:80]}")
        
        # Summary
        print("\n" + "=" * 70)
        print("OPTIMIZATION SUMMARY")
        print("=" * 70)
        
        if self.metrics:
            # Sort by false positive rate
            sorted_models = sorted(self.metrics.items(), key=lambda x: x[1]['false_positive_rate'])
            
            print("\nModels ranked by False Positive Rate:")
            for i, (name, metrics) in enumerate(sorted_models, 1):
                print(f"  {i}. {name.upper():20s} FPR={metrics['false_positive_rate']:.2%} "
                      f"Recall={metrics['recall']:.2%} F1={metrics['f1']:.4f}")
            
            # Best balanced model
            best_model = min(self.metrics.items(), 
                           key=lambda x: abs(x[1]['false_positive_rate'] - target_fpr) - x[1]['recall'])
            print(f"\nBest balanced model: {best_model[0].upper()}")
            
            # Save results
            results_file = Path("model_optimization_results.json")
            output = {
                "target_fpr": target_fpr,
                "test_samples": n_samples,
                "models": self.metrics,
                "thresholds": self.optimized_thresholds,
            }
            with open(results_file, "w") as f:
                json.dump(output, f, indent=2)
            print(f"\nResults saved to {results_file}")


def main():
    """Main optimization pipeline."""
    
    print("\n" + "=" * 70)
    print("THREAT SENTINEL - OPTIMIZED MODEL TESTING".center(70))
    print("Focus: Reducing False Positive Rate".center(70))
    print("=" * 70)
    
    model_dir = "models"
    
    tester = OptimizedModelTester(model_dir=model_dir, n_features=115)
    tester.load_models()
    
    # Test with target FPR of 10%
    tester.run_optimized_tests(n_samples=1000, target_fpr=0.10)


if __name__ == "__main__":
    main()
