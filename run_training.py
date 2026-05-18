"""Entry-point: train the churn model and save artifacts.

Usage::

    python run_training.py
"""

from __future__ import annotations

from src.train_model import save_artifacts, train_and_select


def main() -> None:
    pipeline, best_name, all_metrics = train_and_select()
    model_path = save_artifacts(pipeline, best_name, all_metrics)
    print("=" * 60)
    print(f"Best model: {best_name}")
    print(f"Model artifact: {model_path}")
    print("Metrics summary:")
    for name, m in all_metrics.items():
        marker = "*" if name == best_name else " "
        print(
            f"  {marker} {name:>20s} | "
            f"AUC={m['roc_auc']:.4f}  F1={m['f1']:.4f}  "
            f"Acc={m['accuracy']:.4f}"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
