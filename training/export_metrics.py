"""
Export model metrics in the format expected by the threat-sentinel dashboard.

Creates model_metrics.json with the ModelMetrics[] structure.
"""

import json
from datetime import datetime
from pathlib import Path
from evaluate import ModelEvaluator


def export_metrics_to_dashboard(
    data_dir: str,
    model_dir: str = "models",
    output_file: str = "model_metrics.json",
    subsample_frac: float = 0.5,
) -> None:
    """
    Train/evaluate all models and export metrics.
    
    The exported JSON matches the ModelMetrics[] shape expected by:
    src/lib/models.ts → ModelMetrics interface
    """
    
    print("=" * 70)
    print("EXPORTING METRICS FOR DASHBOARD")
    print("=" * 70)
    
    # Evaluate all models
    evaluator = ModelEvaluator(data_dir, model_dir=model_dir, subsample_frac=subsample_frac)
    evaluator.load_models()
    evaluator.evaluate_all()
    
    # Build metrics JSON
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
    
    # Write to file
    output_path = Path(output_file)
    with open(output_path, "w") as f:
        json.dump(metrics_list, f, indent=2)
    
    print(f"\n✓ Metrics exported to {output_file}")
    print(f"\nMetrics summary:")
    for entry in metrics_list:
        print(f"\n  {entry['model_id']}:")
        print(f"    Accuracy:        {entry['accuracy']:.4f}")
        print(f"    Precision:       {entry['precision']:.4f}")
        print(f"    Recall:          {entry['recall']:.4f}")
        print(f"    F1:              {entry['f1']:.4f}")


if __name__ == "__main__":
    data_dir = r"C:\Users\Tejar\Downloads\archive"
    export_metrics_to_dashboard(data_dir)
