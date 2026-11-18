"""
Test trained models with synthetic data.

This script loads all trained models and runs predictions on synthetic test data.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import tensorflow as tf


class ModelTester:
    """Test trained models with synthetic data."""
    
    def __init__(self, model_dir: str = "models", n_features: int = 115):
        self.model_dir = Path(model_dir)
        self.n_features = n_features
        self.models = {}
        self.metrics = {}
        self.scaler_params = None
    
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
            "autoencoder": "autoencoder.h5",
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
                print(f"  [ERROR] Failed to load {name}: {e}")
    
    def create_synthetic_data(self, n_samples: int = 500, n_benign: int = None) -> tuple:
        """
        Create synthetic test data for model evaluation.
        
        Args:
            n_samples: Total number of test samples
            n_benign: Number of benign samples (rest are attacks). If None, uses n_samples/2
        
        Returns:
            (X_test, y_test): Synthetic features and labels
        """
        if n_benign is None:
            n_benign = n_samples // 2
        n_attack = n_samples - n_benign
        
        # Generate benign samples (low variance, lower mean)
        X_benign = np.random.normal(loc=0.2, scale=0.3, size=(n_benign, self.n_features))
        X_benign = np.clip(X_benign, -1, 1)  # Clip to reasonable range
        
        # Generate attack samples (higher variance, different distribution)
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
    
    def test_sklearn_models(self, X_test: np.ndarray, y_test: np.ndarray) -> None:
        """Test scikit-learn models."""
        sklearn_models = ["random_forest", "xgboost", "svm", "isolation_forest"]
        
        for model_name in sklearn_models:
            if model_name not in self.models:
                continue
            
            model = self.models[model_name]
            
            try:
                y_pred = model.predict(X_test)
                
                # For isolation forest, output is -1/1, need to convert to 0/1
                if model_name == "isolation_forest":
                    y_pred = np.where(y_pred == -1, 1, 0)
                
                # Compute metrics with appropriate parameters
                accuracy = accuracy_score(y_test, y_pred)
                
                # Handle binary/multiclass cases for sklearn metrics
                unique_labels = np.unique(np.hstack([y_test, y_pred]))
                if len(unique_labels) == 2:
                    precision = precision_score(y_test, y_pred, zero_division=0)
                    recall = recall_score(y_test, y_pred, zero_division=0)
                    f1 = f1_score(y_test, y_pred, zero_division=0)
                    avg_type = None
                else:
                    precision = precision_score(y_test, y_pred, zero_division=0, average='weighted')
                    recall = recall_score(y_test, y_pred, zero_division=0, average='weighted')
                    f1 = f1_score(y_test, y_pred, zero_division=0, average='weighted')
                    avg_type = 'weighted'
                
                cm = confusion_matrix(y_test, y_pred)
                if cm.size == 4:
                    tn, fp, fn, tp = cm.ravel()
                    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
                else:
                    fpr = fnr = 0
                
                self.metrics[model_name] = {
                    "accuracy": float(accuracy),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                    "false_positive_rate": float(fpr),
                    "false_negative_rate": float(fnr),
                }
                
                print(f"\n  {model_name.upper()}")
                print(f"    Accuracy:  {accuracy:.4f}")
                print(f"    Precision: {precision:.4f}")
                print(f"    Recall:    {recall:.4f}")
                print(f"    F1 Score:  {f1:.4f}")
                print(f"    FPR:       {fpr:.4f}")
                print(f"    FNR:       {fnr:.4f}")
                
            except Exception as e:
                print(f"\n  {model_name.upper()}: ERROR - {e}")
    
    def test_neural_models(self, X_test: np.ndarray, y_test: np.ndarray) -> None:
        """Test neural network models."""
        
        # Test Autoencoder
        if "autoencoder" in self.models:
            try:
                model = self.models["autoencoder"]
                
                # Compute reconstruction error
                X_recon = model.predict(X_test, verbose=0)
                mse = np.mean((X_test - X_recon) ** 2, axis=1)
                
                # Use threshold-based detection
                threshold = np.percentile(mse, 75)  # Top 25% as anomalies
                y_pred = (mse > threshold).astype(int)
                
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                
                self.metrics["autoencoder"] = {
                    "accuracy": float(accuracy),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                }
                
                print(f"\n  AUTOENCODER")
                print(f"    Accuracy:  {accuracy:.4f}")
                print(f"    Precision: {precision:.4f}")
                print(f"    Recall:    {recall:.4f}")
                print(f"    F1 Score:  {f1:.4f}")
                
            except Exception as e:
                print(f"\n  AUTOENCODER: ERROR - {type(e).__name__}: {str(e)[:80]}")
        
        # Test CNN
        if "cnn" in self.models:
            try:
                model = self.models["cnn"]
                
                # CNN expects (window_size, n_features) shape
                window_size = 10
                n_full_windows = len(X_test) // window_size
                
                if n_full_windows > 0:
                    # Trim to multiple of window_size
                    X_trimmed = X_test[:n_full_windows * window_size]
                    y_trimmed = y_test[:n_full_windows * window_size]
                    
                    # Reshape into sequences (batch, window_size, features)
                    X_test_cnn = X_trimmed.reshape(n_full_windows, window_size, self.n_features)
                    y_test_cnn = y_trimmed.reshape(n_full_windows, window_size)[:, -1]  # Last label
                    
                    y_pred_proba = model.predict(X_test_cnn, verbose=0)
                    y_pred = (y_pred_proba > 0.5).astype(int).flatten()[:len(y_test_cnn)]
                    
                    accuracy = accuracy_score(y_test_cnn, y_pred)
                    precision = precision_score(y_test_cnn, y_pred, zero_division=0)
                    recall = recall_score(y_test_cnn, y_pred, zero_division=0)
                    f1 = f1_score(y_test_cnn, y_pred, zero_division=0)
                    
                    self.metrics["cnn"] = {
                        "accuracy": float(accuracy),
                        "precision": float(precision),
                        "recall": float(recall),
                        "f1": float(f1),
                    }
                    
                    print(f"\n  CNN")
                    print(f"    Accuracy:  {accuracy:.4f}")
                    print(f"    Precision: {precision:.4f}")
                    print(f"    Recall:    {recall:.4f}")
                    print(f"    F1 Score:  {f1:.4f}")
                else:
                    print(f"\n  CNN: Insufficient test samples (need at least {window_size})")
                
            except Exception as e:
                print(f"\n  CNN: ERROR - {type(e).__name__}: {str(e)[:80]}")
        
        # Test CNN-LSTM
        if "cnn_lstm" in self.models:
            try:
                model = self.models["cnn_lstm"]
                
                # Reshape for CNN-LSTM (sequence of timesteps)
                window_size = 10
                n_full_windows = len(X_test) // window_size
                
                if n_full_windows > 0:
                    # Trim to multiple of window_size
                    X_trimmed = X_test[:n_full_windows * window_size]
                    y_trimmed = y_test[:n_full_windows * window_size]
                    
                    # Reshape into sequences
                    X_test_seq = X_trimmed.reshape(n_full_windows, window_size, self.n_features)
                    y_test_seq = y_trimmed.reshape(n_full_windows, window_size)[:, -1]  # Last label of sequence
                    
                    # Try different input shapes
                    try:
                        # Try with channels last
                        X_test_seq_ch = X_test_seq[..., np.newaxis]
                        y_pred_proba = model.predict(X_test_seq_ch, verbose=0)
                    except:
                        # Try original shape
                        y_pred_proba = model.predict(X_test_seq, verbose=0)
                    
                    y_pred = (y_pred_proba > 0.5).astype(int).flatten()[:len(y_test_seq)]
                    
                    accuracy = accuracy_score(y_test_seq, y_pred)
                    precision = precision_score(y_test_seq, y_pred, zero_division=0)
                    recall = recall_score(y_test_seq, y_pred, zero_division=0)
                    f1 = f1_score(y_test_seq, y_pred, zero_division=0)
                    
                    self.metrics["cnn_lstm"] = {
                        "accuracy": float(accuracy),
                        "precision": float(precision),
                        "recall": float(recall),
                        "f1": float(f1),
                    }
                    
                    print(f"\n  CNN-LSTM")
                    print(f"    Accuracy:  {accuracy:.4f}")
                    print(f"    Precision: {precision:.4f}")
                    print(f"    Recall:    {recall:.4f}")
                    print(f"    F1 Score:  {f1:.4f}")
                else:
                    print(f"\n  CNN-LSTM: Insufficient test samples (need at least {window_size})")
                    
            except Exception as e:
                print(f"\n  CNN-LSTM: ERROR - {type(e).__name__}: {str(e)[:80]}")
    
    def run_tests(self, n_samples: int = 500) -> None:
        """Run all model tests."""
        # Create synthetic data
        print("\n" + "=" * 70)
        print("GENERATING SYNTHETIC TEST DATA")
        print("=" * 70)
        X_test, y_test = self.create_synthetic_data(n_samples)
        
        # Test models
        print("\n" + "=" * 70)
        print("TESTING MODELS")
        print("=" * 70)
        
        self.test_sklearn_models(X_test, y_test)
        self.test_neural_models(X_test, y_test)
        
        # Summary
        print("\n" + "=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)
        print(f"\nTested {len(self.models)} models")
        print(f"Generated metrics for {len(self.metrics)} models")
        
        # Save metrics to JSON
        metrics_file = Path("model_test_results.json")
        with open(metrics_file, "w") as f:
            json.dump(self.metrics, f, indent=2)
        print(f"\nMetrics saved to {metrics_file}")
        
        # Print detailed summary table
        if self.metrics:
            print("\n" + "=" * 70)
            print("DETAILED RESULTS")
            print("=" * 70)
            df = pd.DataFrame(self.metrics).T
            print("\n" + df.to_string())


def main():
    """Main test pipeline."""
    
    print("\n")
    print("=" * 70)
    print("THREAT SENTINEL - MODEL TEST".center(70))
    print("=" * 70)
    print()
    
    model_dir = "models"
    
    tester = ModelTester(model_dir=model_dir, n_features=115)
    tester.load_models()
    tester.run_tests(n_samples=500)


if __name__ == "__main__":
    main()
