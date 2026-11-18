"""
Test trained models on REAL N-BaIoT dataset.

This script evaluates models on the actual data they were trained for:
N-BaIoT IoT Botnet dataset with real benign and attack traffic.

Dataset structure expected:
- CSV files with names containing "benign", "mirai", or "gafgyt"
- 115 features per record (network traffic statistics)
- Labels determined by filename
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import json
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)
import tensorflow as tf


class RealDataTester:
    """Test models on real N-BaIoT dataset."""
    
    def __init__(self, model_dir: str = "models", data_dir: str = None):
        self.model_dir = Path(model_dir)
        self.data_dir = Path(data_dir) if data_dir else Path(r"C:\Users\Tejar\Downloads\archive")
        self.models = {}
        self.data = {}
        self.metrics = {}
    
    def check_data_availability(self) -> bool:
        """Check if N-BaIoT data is available."""
        if not self.data_dir.exists():
            print(f"\n[ERROR] Data directory not found: {self.data_dir}")
            print("\nTo use this script, download N-BaIoT dataset:")
            print("  1. Visit: https://www.kaggle.com/datasets/mrx7014/nbaiot-dataset")
            print("  2. Download and extract to: " + str(self.data_dir))
            print("  3. Dataset should contain CSV files:")
            print("     - *_benign.csv (benign traffic)")
            print("     - *_mirai.csv or *_gafgyt.csv (attack traffic)")
            return False
        
        csv_files = list(self.data_dir.glob("*.csv"))
        if not csv_files:
            print(f"\n[ERROR] No CSV files found in {self.data_dir}")
            return False
        
        print(f"\n[OK] Found {len(csv_files)} CSV files")
        print("Sample files:")
        for f in csv_files[:5]:
            print(f"  - {f.name}")
        
        return True
    
    def load_nbaiot_data(self, subsample_frac: float = 1.0) -> bool:
        """Load N-BaIoT dataset from CSV files."""
        
        print("\n" + "=" * 70)
        print("LOADING N-BaIoT DATASET")
        print("=" * 70)
        
        csv_files = sorted(self.data_dir.glob("*.csv"))
        all_data = []
        
        for csv_file in csv_files:
            filename = csv_file.name
            
            # Skip metadata files
            if filename in ["data_summary.csv", "device_info.csv", "features.csv"]:
                continue
            
            # Determine label
            if "benign" in filename.lower():
                label = 0
                label_name = "BENIGN"
            elif "mirai" in filename.lower():
                label = 1
                label_name = "MIRAI"
            elif "gafgyt" in filename.lower():
                label = 1
                label_name = "GAFGYT"
            else:
                continue
            
            try:
                print(f"  Loading {filename}...", end=" ")
                df = pd.read_csv(csv_file)
                
                # Subsample if requested
                if subsample_frac < 1.0:
                    df = df.sample(frac=subsample_frac, random_state=42)
                
                df["label"] = label
                all_data.append(df)
                print(f"OK ({len(df)} records, {label_name})")
                
            except Exception as e:
                print(f"ERROR: {e}")
                continue
        
        if not all_data:
            print("\n[ERROR] No data loaded")
            return False
        
        # Combine data
        data = pd.concat(all_data, ignore_index=True)
        
        print(f"\n  Total records: {len(data):,}")
        print(f"  Benign: {(data['label'] == 0).sum():,}")
        print(f"  Attack: {(data['label'] == 1).sum():,}")
        print(f"  Features: {len(data.columns) - 1}")
        
        # Prepare X and y
        X = data.drop("label", axis=1).values
        y = data["label"].values
        
        # Standardize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.data = {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "scaler": scaler,
        }
        
        print(f"\n  Train set: {len(X_train):,} samples")
        print(f"  Test set: {len(X_test):,} samples")
        
        return True
    
    def load_models(self) -> None:
        """Load trained models."""
        print("\n" + "=" * 70)
        print("LOADING TRAINED MODELS")
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
                print(f"  [MISSING] {name}")
                continue
            
            try:
                if filename.endswith(".pkl"):
                    with open(filepath, "rb") as f:
                        self.models[name] = pickle.load(f)
                else:  # .h5
                    self.models[name] = tf.keras.models.load_model(str(filepath))
                
                print(f"  [OK] Loaded {name}")
            except Exception as e:
                print(f"  [ERROR] {name}: {type(e).__name__}")
    
    def test_sklearn_model(self, model_name: str) -> dict:
        """Test a scikit-learn model."""
        model = self.models[model_name]
        X_test = self.data["X_test"]
        y_test = self.data["y_test"]
        
        try:
            y_pred = model.predict(X_test)
            
            # Handle isolation forest output (-1/1 instead of 0/1)
            if model_name == "isolation_forest":
                y_pred = np.where(y_pred == -1, 1, 0)
            
            # Get probabilities if available
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, y_proba)
            elif hasattr(model, 'score_samples'):
                y_scores = model.score_samples(X_test)
                auc = roc_auc_score(y_test, -y_scores)  # Negate for IF
            else:
                auc = None
            
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
                tn = fp = fn = tp = fpr = fnr = 0
            
            return {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "auc": float(auc) if auc else None,
                "fpr": float(fpr),
                "fnr": float(fnr),
                "tp": int(tp),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
            }
        
        except Exception as e:
            print(f"    Error: {e}")
            return None
    
    def test_neural_model(self, model_name: str) -> dict:
        """Test a neural network model."""
        model = self.models[model_name]
        X_test = self.data["X_test"]
        y_test = self.data["y_test"]
        
        try:
            if model_name == "cnn":
                # CNN expects (batch, window_size, features)
                window_size = 10
                n_windows = len(X_test) // window_size
                if n_windows > 0:
                    X_reshaped = X_test[:n_windows * window_size].reshape(n_windows, window_size, X_test.shape[1])
                    y_proba = model.predict(X_reshaped, verbose=0).flatten()
                    y_pred = (y_proba > 0.5).astype(int)
                    y_test_trimmed = y_test[:len(y_pred)]
                else:
                    return None
            
            elif model_name == "cnn_lstm":
                # CNN-LSTM expects sequences
                window_size = 10
                n_windows = len(X_test) // window_size
                if n_windows > 0:
                    X_reshaped = X_test[:n_windows * window_size].reshape(n_windows, window_size, X_test.shape[1])
                    y_proba = model.predict(X_reshaped, verbose=0).flatten()
                    y_pred = (y_proba > 0.5).astype(int)
                    y_test_trimmed = y_test[:len(y_pred)]
                else:
                    return None
            else:
                return None
            
            # Compute metrics
            accuracy = accuracy_score(y_test_trimmed, y_pred)
            precision = precision_score(y_test_trimmed, y_pred, zero_division=0)
            recall = recall_score(y_test_trimmed, y_pred, zero_division=0)
            f1 = f1_score(y_test_trimmed, y_pred, zero_division=0)
            
            try:
                auc = roc_auc_score(y_test_trimmed, y_proba)
            except:
                auc = None
            
            cm = confusion_matrix(y_test_trimmed, y_pred)
            if cm.size == 4:
                tn, fp, fn, tp = cm.ravel()
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            else:
                tn = fp = fn = tp = fpr = fnr = 0
            
            return {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "auc": float(auc) if auc else None,
                "fpr": float(fpr),
                "fnr": float(fnr),
                "tp": int(tp),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
            }
        
        except Exception as e:
            print(f"    Error: {e}")
            return None
    
    def run_tests(self) -> None:
        """Run all tests on real data."""
        
        if not self.check_data_availability():
            return
        
        if not self.load_nbaiot_data():
            return
        
        self.load_models()
        
        if not self.models:
            print("\n[ERROR] No models loaded")
            return
        
        # Test models
        print("\n" + "=" * 70)
        print("TESTING ON REAL N-BaIoT DATA")
        print("=" * 70)
        
        for model_name in self.models.keys():
            print(f"\n  Testing {model_name.upper()}...", end=" ")
            
            if model_name in ["cnn", "cnn_lstm"]:
                result = self.test_neural_model(model_name)
            else:
                result = self.test_sklearn_model(model_name)
            
            if result:
                self.metrics[model_name] = result
                print("OK")
                print(f"    Accuracy:  {result['accuracy']:.2%}")
                print(f"    Precision: {result['precision']:.2%}")
                print(f"    Recall:    {result['recall']:.2%}")
                print(f"    F1 Score:  {result['f1']:.4f}")
                print(f"    AUC:       {result['auc']:.4f if result['auc'] else 'N/A'}")
                print(f"    FPR:       {result['fpr']:.2%}")
                print(f"    FNR:       {result['fnr']:.2%}")
        
        # Summary
        print("\n" + "=" * 70)
        print("REAL DATA TEST SUMMARY")
        print("=" * 70)
        
        if self.metrics:
            df = pd.DataFrame(self.metrics).T
            print("\n" + df.to_string())
            
            # Best model
            best = max(self.metrics.items(), key=lambda x: x[1]['f1'])
            print(f"\nBest model: {best[0].upper()} (F1={best[1]['f1']:.4f})")
            
            # Save results
            with open("real_data_test_results.json", "w") as f:
                json.dump(self.metrics, f, indent=2, default=str)
            print("\nResults saved to real_data_test_results.json")


def main():
    print("\n" + "=" * 70)
    print("THREAT SENTINEL - REAL DATA EVALUATION".center(70))
    print("=" * 70)
    
    tester = RealDataTester()
    tester.run_tests()


if __name__ == "__main__":
    main()
