# Threat Sentinel - ML Training Kit

Train 7 ML models on the N-BaIoT dataset for network intrusion detection.

## Models

1. **Random Forest** - Ensemble of decision trees
2. **XGBoost** - Gradient boosted trees
3. **SVM (RBF)** - Support Vector Machine
4. **Isolation Forest** - Unsupervised anomaly detection
5. **Autoencoder** - Reconstruction error-based detection
6. **CNN** - 1D Convolutional Neural Network
7. **CNN-LSTM** - Hybrid CNN + LSTM (deep learning)

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train Models

```bash
python train_models.py
```

This will:
- Load CSV files from `C:\Users\Tejar\Downloads\archive`
- Balance benign and attack data
- Split into train/test sets
- Standardize features
- Train all 7 models
- Save models to `models/` directory

Training takes ~5-10 minutes on a modern CPU.

### 3. Evaluate Models

```bash
python evaluate.py
```

Computes metrics for each model:
- Accuracy
- Precision
- Recall
- F1 Score
- Detection Rate
- False Positive Rate

### 4. Export Metrics for Dashboard

```bash
python export_metrics.py
```

Generates `model_metrics.json` with the format expected by the Threat Sentinel dashboard.

## Usage on Windows

```powershell
# Activate virtual environment (if using venv)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Run training pipeline
python train_models.py
python evaluate.py
python export_metrics.py
```

## Dataset

The N-BaIoT dataset contains network traffic features from 9 IoT devices:
- **Benign traffic**: Normal device behavior
- **Mirai attacks**: DDoS malware (ACK flood, SYN flood, UDP flood, UDP plain)
- **Gafgyt attacks**: Another DDoS malware (TCP, UDP, scan, combo, junk)

Each record has 115 statistical features extracted from network flows.

## Integration with Dashboard

1. After training, copy `model_metrics.json` to the project root
2. The dashboard will automatically load real metrics instead of placeholders
3. Or set up the optional FastAPI server for live metric serving:

```bash
python serve_metrics.py
```

Then update `src/lib/models.ts`:
```typescript
const MODEL_METRICS_API_URL = "http://localhost:8000/model_metrics";
```

## Files

- `data_prep.py` - Data loading, preprocessing, windowing
- `train_models.py` - Model training
- `evaluate.py` - Model evaluation and metric computation
- `export_metrics.py` - Export metrics in dashboard format
- `requirements.txt` - Python dependencies
- `models/` - Trained model files (created after training)

## Output

After running the pipeline:

```
models/
  ├── random_forest.pkl
  ├── xgboost.pkl
  ├── svm.pkl
  ├── isolation_forest.pkl
  ├── autoencoder.h5
  ├── cnn.h5
  └── cnn_lstm.h5

model_metrics.json  # Dashboard metrics
```

## Performance Notes

- Training uses 50% of the dataset for faster iteration
- Models achieve 90%+ accuracy on attack detection
- CNN-LSTM is the most accurate but slowest to train
- Classical models (RF, XGB, SVM) train in seconds
- Unsupervised models (IF, Autoencoder) run on CPU

## Next Steps

1. Copy `model_metrics.json` to the dashboard project
2. Update `src/lib/models.ts` to use real metrics
3. Deploy the dashboard with live model predictions
4. Monitor model performance in production

## Troubleshooting

**Models not loading?**
- Ensure TensorFlow is installed: `pip install tensorflow`
- Check model file paths in `models/` directory
- Verify scikit-learn and XGBoost are installed

**Data loading fails?**
- Verify archive path: `C:\Users\Tejar\Downloads\archive`
- Ensure CSV files exist in the directory
- Check file permissions

**Out of memory?**
- Reduce `subsample_frac` in training scripts (e.g., 0.25)
- Reduce batch size in neural network training
- Use GPU with TensorFlow: `pip install tensorflow[and-cuda]`
