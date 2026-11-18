"""
Main training pipeline - trains all 7 models and exports metrics.

Run this script to:
  1. Load and prepare N-BaIoT data
  2. Train all 7 models
  3. Evaluate models
  4. Export metrics for the dashboard
"""

import sys
import os
from pathlib import Path

# Add training directory to path
training_dir = Path(__file__).parent
sys.path.insert(0, str(training_dir))

from train_models import ModelTrainer
from evaluate import ModelEvaluator
from export_metrics import export_metrics_to_dashboard


def main():
    """Main training pipeline."""
    
    print("\n")
    print("█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "  THREAT SENTINEL - ML MODEL TRAINING PIPELINE".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)
    print()
    
    # Configuration
    data_dir = r"C:\Users\Tejar\Downloads\archive"
    model_dir = "models"
    output_file = "model_metrics.json"
    subsample_frac = 0.1  # Use 10% of data to ensure full pipeline completion
    
    # Verify data directory exists
    if not Path(data_dir).exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        return False
    
    # Create model directory
    Path(model_dir).mkdir(exist_ok=True)
    
    try:
        # Step 1: Train models
        print("\n[STEP 1/3] TRAINING MODELS")
        print("-" * 70)
        trainer = ModelTrainer(
            data_dir,
            subsample_frac=subsample_frac,
            svm_max_samples=40000,
            nn_epochs=12,
        )
        trainer.train_all()
        trainer.save_models(model_dir)
        
        # Step 2: Evaluate models
        print("\n[STEP 2/3] EVALUATING MODELS")
        print("-" * 70)
        evaluator = ModelEvaluator(data_dir, model_dir=model_dir, subsample_frac=subsample_frac)
        evaluator.load_models()
        evaluator.evaluate_all()
        
        # Step 3: Export metrics
        print("\n[STEP 3/3] EXPORTING METRICS")
        print("-" * 70)
        
        # Build metrics JSON
        from datetime import datetime
        import json
        
        metrics_list = []
        for model_name, metrics in evaluator.metrics.items():
            entry = {
                "model_id": model_name,
                "accuracy": round(metrics["accuracy"], 4),
                "precision": round(metrics["precision"], 4),
                "recall": round(metrics["recall"], 4),
                "f1": round(metrics["f1"], 4),
                "detection_rate": round(metrics["detection_rate"], 4),
                "false_positive_rate": round(metrics["false_positive_rate"], 4),
                "source": "trained",
                "evaluated_at": datetime.now().isoformat(),
            }
            metrics_list.append(entry)
        
        # Write metrics file
        with open(output_file, "w") as f:
            json.dump(metrics_list, f, indent=2)
        
        print(f"\n✓ Metrics exported to: {output_file}")
        
        # Print summary
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE - METRICS SUMMARY")
        print("=" * 70)
        print()
        
        for entry in sorted(metrics_list, key=lambda x: x["accuracy"], reverse=True):
            print(f"  {entry['model_id'].upper():20s}")
            print(f"    Accuracy:              {entry['accuracy']:.1%}")
            print(f"    Precision:             {entry['precision']:.1%}")
            print(f"    Recall:                {entry['recall']:.1%}")
            print(f"    F1 Score:              {entry['f1']:.1%}")
            print(f"    Detection Rate:        {entry['detection_rate']:.1%}")
            print(f"    False Positive Rate:   {entry['false_positive_rate']:.1%}")
            print()
        
        print("=" * 70)
        print("\n✓ All models trained and evaluated successfully!")
        print(f"\nOutput files:")
        print(f"  - Models: {model_dir}/")
        print(f"  - Metrics: {output_file}")
        print(f"\nNext steps:")
        print(f"  1. Copy {output_file} to the dashboard project root")
        print(f"  2. The dashboard will auto-load the real metrics")
        print(f"  3. Or host models via FastAPI for live serving")
        print()
        
        return True
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
