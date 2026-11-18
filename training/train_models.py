"""
Train 7 ML models for network intrusion detection on N-BaIoT dataset.

Models:
  1. Random Forest (classical)
  2. XGBoost (classical)
  3. SVM with RBF kernel (classical)
  4. Isolation Forest (unsupervised)
  5. Autoencoder (unsupervised)
  6. CNN (1D convolutional)
  7. CNN-LSTM (hybrid)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import json

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential, Model
from tensorflow.keras.optimizers import Adam

from data_prep import prepare_data, prepare_sequences


class ModelTrainer:
    """Container for training and evaluating all models."""
    
    def __init__(
        self,
        data_dir: str,
        subsample_frac: float = 0.1,
        svm_max_samples: int = 40000,
        nn_epochs: int = 12,
    ):
        self.data_dir = data_dir
        self.subsample_frac = subsample_frac
        self.svm_max_samples = svm_max_samples
        self.nn_epochs = nn_epochs
        self.models = {}
        self.histories = {}
        
        # Prepare data
        print("=" * 70)
        print("PREPARING DATA")
        print("=" * 70)
        self.data = prepare_data(data_dir, subsample_frac=subsample_frac)
        self.data_seq = prepare_sequences(self.data, window_size=10)
    
    def train_random_forest(self) -> None:
        """Train Random Forest classifier."""
        print("\n" + "=" * 70)
        print("TRAINING: Random Forest")
        print("=" * 70)
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(self.data["X_train"], self.data["y_train"])
        
        self.models["random_forest"] = model
        print("Random Forest training complete")
    
    def train_xgboost(self) -> None:
        """Train XGBoost classifier."""
        print("\n" + "=" * 70)
        print("TRAINING: XGBoost")
        print("=" * 70)
        
        model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        model.fit(self.data["X_train"], self.data["y_train"])
        
        self.models["xgboost"] = model
        print("XGBoost training complete")
    
    def train_svm(self) -> None:
        """Train SVM with RBF kernel."""
        print("\n" + "=" * 70)
        print("TRAINING: SVM (RBF)")
        print("=" * 70)
        
        X_train = self.data["X_train"]
        y_train = self.data["y_train"]

        # Cap SVM sample count to avoid very long training times.
        if len(X_train) > self.svm_max_samples:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(X_train), size=self.svm_max_samples, replace=False)
            X_train = X_train[idx]
            y_train = y_train[idx]
            print(f"SVM using capped sample size: {len(X_train)}")

        model = SVC(kernel="rbf", gamma="scale", probability=True, random_state=42)
        model.fit(X_train, y_train)
        
        self.models["svm"] = model
        print("SVM training complete")
    
    def train_isolation_forest(self) -> None:
        """Train Isolation Forest (unsupervised anomaly detector)."""
        print("\n" + "=" * 70)
        print("TRAINING: Isolation Forest")
        print("=" * 70)
        
        # Train only on benign data
        benign_indices = self.data["y_train"] == 0
        X_benign = self.data["X_train"][benign_indices]
        
        model = IsolationForest(
            contamination=0.01,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_benign)
        
        self.models["isolation_forest"] = model
        print("Isolation Forest training complete")
    
    def train_autoencoder(self) -> None:
        """Train Autoencoder (unsupervised reconstruction error detector)."""
        print("\n" + "=" * 70)
        print("TRAINING: Autoencoder")
        print("=" * 70)
        
        # Train only on benign data
        benign_indices = self.data["y_train"] == 0
        X_benign = self.data["X_train"][benign_indices]
        
        input_dim = X_benign.shape[1]
        
        # Build autoencoder
        model = Sequential([
            layers.Dense(64, activation="relu", input_dim=input_dim),
            layers.Dense(32, activation="relu"),
            layers.Dense(16, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(input_dim, activation="linear"),
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
        
        # Train
        history = model.fit(
            X_benign,
            X_benign,
            epochs=self.nn_epochs,
            batch_size=32,
            validation_split=0.2,
            verbose=0,
        )
        
        self.models["autoencoder"] = model
        self.histories["autoencoder"] = history
        print("Autoencoder training complete")
    
    def train_cnn(self) -> None:
        """Train 1D CNN model."""
        print("\n" + "=" * 70)
        print("TRAINING: CNN (1D Convolutional)")
        print("=" * 70)
        
        input_shape = self.data_seq["X_train"].shape[1:]  # (window_size, n_features)
        
        model = Sequential([
            layers.Conv1D(64, kernel_size=3, activation="relu", input_shape=input_shape),
            layers.MaxPooling1D(pool_size=2),
            layers.Conv1D(32, kernel_size=3, activation="relu"),
            layers.MaxPooling1D(pool_size=2),
            layers.Flatten(),
            layers.Dense(32, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(1, activation="sigmoid"),
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        
        # Train
        history = model.fit(
            self.data_seq["X_train"],
            self.data_seq["y_train"],
            epochs=self.nn_epochs,
            batch_size=32,
            validation_split=0.2,
            verbose=0,
        )
        
        self.models["cnn"] = model
        self.histories["cnn"] = history
        print("CNN training complete")
    
    def train_cnn_lstm(self) -> None:
        """Train hybrid CNN-LSTM model."""
        print("\n" + "=" * 70)
        print("TRAINING: CNN-LSTM (Hybrid)")
        print("=" * 70)
        
        input_shape = self.data_seq["X_train"].shape[1:]  # (window_size, n_features)
        
        model = Sequential([
            layers.Conv1D(64, kernel_size=3, activation="relu", input_shape=input_shape),
            layers.MaxPooling1D(pool_size=2),
            layers.LSTM(32, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(16, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        
        # Train
        history = model.fit(
            self.data_seq["X_train"],
            self.data_seq["y_train"],
            epochs=self.nn_epochs,
            batch_size=32,
            validation_split=0.2,
            verbose=0,
        )
        
        self.models["cnn_lstm"] = model
        self.histories["cnn_lstm"] = history
        print("CNN-LSTM training complete")
    
    def train_all(self) -> None:
        """Train all 7 models."""
        self.train_random_forest()
        self.train_xgboost()
        self.train_svm()
        self.train_isolation_forest()
        self.train_autoencoder()
        self.train_cnn()
        self.train_cnn_lstm()
        
        print("\n" + "=" * 70)
        print("ALL MODELS TRAINED")
        print("=" * 70)
    
    def save_models(self, output_dir: str = "models") -> None:
        """Save trained models and the StandardScaler."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Save trained models
        for name, model in self.models.items():
            if isinstance(model, keras.Model):
                model.save(str(output_path / f"{name}.h5"))
            else:
                with open(output_path / f"{name}.pkl", "wb") as f:
                    pickle.dump(model, f)

        # Save the StandardScaler used during training
        with open(output_path / "scaler.pkl", "wb") as f:
            pickle.dump(self.data["scaler"], f)

        print(f"\nModels and scaler saved to {output_dir}/")
        for name, model in self.models.items():
            if isinstance(model, keras.Model):
                model.save(str(output_path / f"{name}.h5"))
            else:
                with open(output_path / f"{name}.pkl", "wb") as f:
                    pickle.dump(model, f)
        
        print(f"\nModels saved to {output_dir}/")


def main():
    """Main training script."""
    data_dir = r"C:\Users\Tejar\Downloads\archive"
    
    trainer = ModelTrainer(data_dir, subsample_frac=0.1, svm_max_samples=40000, nn_epochs=12)
    trainer.train_all()
    trainer.save_models()
    
    print("\n✓ Training complete! Models saved.")


if __name__ == "__main__":
    main()
