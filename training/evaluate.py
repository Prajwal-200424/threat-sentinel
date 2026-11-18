"""
Evaluate trained models and compute metrics.

Computes: accuracy, precision, recall, F1, detection_rate, false_positive_rate
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import json

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import tensorflow as tf

from data_prep import prepare_data, prepare_sequences


class ModelEvaluator:
    """Evaluate trained models."""
    
    def __init__(self, data_dir: str, model_dir: str = "models", subsample_frac: float = 0.5):
        self.data_dir = data_dir
        self.model_dir = Path(model_dir)
        self.subsample_frac = subsample_frac
        
        # Prepare data
        print("=" * 70)
        print("PREPARING DATA FOR EVALUATION")
        print("=" * 70)
        self.data = prepare_data(data_dir, subsample_frac=subsample_frac)
        self.data_seq = prepare_sequences(self.data, window_size=10)
        
        self.models = {}
        self.metrics = {}
    
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
                print(f"  ✗ {name}: {filename} not found")
                continue
            
            try:
                if filename.endswith(".pkl"):
                    with open(filepath, "rb") as f:
                        self.models[name] = pickle.load(f)
                else:  # .h5
                    self.models[name] = tf.keras.models.load_model(str(filepath))
                
                print(f"  ✓ Loaded {name}")
            except Exception as e:
                print(f"  ✗ Failed to load {name}: {e}")
    
    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray = None) -> dict:
        """
        Compute evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels (0 or 1)
            y_pred_proba: Predicted probabilities (optional)
        
        Returns:
            dict with metrics normalized to 0-1 range
        """
        # Ensure binary predictions
        y_pred = (y_pred > 0.5).astype(int) if y_pred.dtype == np.float32 or y_pred.dtype == np.float64 else y_pred
        
        # Compute metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # detection_rate
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # false_positive_rate
        
        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "detection_rate": float(recall),
            "false_positive_rate": float(fpr),
        }
    
    def evaluate_random_forest(self) -> None:
        """Evaluate Random Forest."""
        print("\nEvaluating Random Forest...")
        model = self.models.get("random_forest")
        if model is None:
            return
        
        y_pred = model.predict(self.data["X_test"])
        metrics = self.compute_metrics(self.data["y_test"], y_pred)
        self.metrics["random_forest"] = metrics
        
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1:        {metrics['f1']:.4f}")
    
    def evaluate_xgboost(self) -> None:
        """Evaluate XGBoost."""
        print("\nEvaluating XGBoost...")
        model = self.models.get("xgboost")
        if model is None:
            return
        
        y_pred = model.predict(self.data["X_test"])
        metrics = self.compute_metrics(self.data["y_test"], y_pred)
        self.metrics["xgboost"] = metrics
        
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1:        {metrics['f1']:.4f}")
    
    def evaluate_svm(self) -> None:
        """Evaluate SVM."""
        print("\nEvaluating SVM...")
        model = self.models.get("svm")
        if model is None:
            return
        
        y_pred = model.predict(self.data["X_test"])
        metrics = self.compute_metrics(self.data["y_test"], y_pred)
        self.metrics["svm"] = metrics
        
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1:        {metrics['f1']:.4f}")
    
    def evaluate_isolation_forest(self) -> None:
        """Evaluate Isolation Forest."""
        print("\nEvaluating Isolation Forest...")
        model = self.models.get("isolation_forest")
        if model is None:
            return
        
        # Predict (-1 = anomaly, 1 = normal)
        y_pred_raw = model.predict(self.data["X_test"])
        # Convert to binary labels (1 = anomaly, 0 = normal)
        y_pred = (y_pred_raw == -1).astype(int)
        
        metrics = self.compute_metrics(self.data["y_test"], y_pred)
        self.metrics["isolation_forest"] = metrics
        
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1:        {metrics['f1']:.4f}")
    
    def evaluate_autoencoder(self) -> None:
        """Evaluate Autoencoder."""
        print("\nEvaluating Autoencoder...")
        model = self.models.get("autoencoder")
        if model is None:
            return
        
        # Compute reconstruction error
        X_test_pred = model.predict(self.data["X_test"], verbose=0)
        reconstruction_error = np.mean(np.square(self.data["X_test"] - X_test_pred), axis=1)
        
        # Threshold at 95th percentile
        threshold = np.percentile(reconstruction_error, 95)
        y_pred = (reconstruction_error > threshold).astype(int)
        
        metrics = self.compute_metrics(self.data["y_test"], y_pred)
        self.metrics["autoencoder"] = metrics
        
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1:        {metrics['f1']:.4f}")
    
    def evaluate_cnn(self) -> None:
        """Evaluate CNN."""
        print("\nEvaluating CNN...")
        model = self.models.get("cnn")
        if model is None:
            return
        
        y_pred_proba = model.predict(self.data_seq["X_test"], verbose=0).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        metrics = self.compute_metrics(self.data_seq["y_test"], y_pred, y_pred_proba)
        self.metrics["cnn"] = metrics
        
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1:        {metrics['f1']:.4f}")
    
    def evaluate_cnn_lstm(self) -> None:
        """Evaluate CNN-LSTM."""
        print("\nEvaluating CNN-LSTM...")
        model = self.models.get("cnn_lstm")
        if model is None:
            return
        
        y_pred_proba = model.predict(self.data_seq["X_test"], verbose=0).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        metrics = self.compute_metrics(self.data_seq["y_test"], y_pred, y_pred_proba)
        self.metrics["cnn_lstm"] = metrics
        
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1:        {metrics['f1']:.4f}")
    
    def evaluate_all(self) -> None:
        """Evaluate all models."""
        print("\n" + "=" * 70)
        print("EVALUATING MODELS")
        print("=" * 70)
        
        self.evaluate_random_forest()
        self.evaluate_xgboost()
        self.evaluate_svm()
        self.evaluate_isolation_forest()
        self.evaluate_autoencoder()
        self.evaluate_cnn()
        self.evaluate_cnn_lstm()
        
        print("\n" + "=" * 70)
        print("EVALUATION COMPLETE")
        print("=" * 70)
    
    def export_metrics(self, output_file: str = "model_metrics.json") -> None:
        """Export metrics to JSON."""
        from datetime import datetime
        
        metrics_list = []
        
        for model_name, metrics in self.metrics.items():
            metrics_list.append({
                "model_id": model_name,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "detection_rate": metrics["detection_rate"],
                "false_positive_rate": metrics["false_positive_rate"],
                "source": "trained",
                "evaluated_at": datetime.now().isoformat(),
            })
        
        with open(output_file, "w") as f:
            json.dump(metrics_list, f, indent=2)
        
        print(f"\nMetrics exported to {output_file}")


def main():
    """Main evaluation script."""
    data_dir = r"C:\Users\Tejar\Downloads\archive"
    
    evaluator = ModelEvaluator(data_dir, model_dir="models", subsample_frac=0.5)
    evaluator.load_models()
    evaluator.evaluate_all()
    evaluator.export_metrics("model_metrics.json")


if __name__ == "__main__":
    main()
