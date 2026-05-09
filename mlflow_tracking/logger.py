"""
logger.py
---------
Centralised MLflow logging utilities used by both the training
and evaluation scripts to keep experiment tracking consistent.
"""

import mlflow
import os


def setup_mlflow(experiment_name: str):
    """
    Configures MLflow tracking URI and sets the active experiment.

    Args:
        experiment_name: Name of the MLflow experiment to log under
    """
    uri = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)
    print(f"[MLflow] Tracking to: {uri}  |  Experiment: {experiment_name}")


def log_training_run(params: dict, metrics: dict, artifacts_dir: str = None):
    """
    Logs a single training run with params, metrics, and optional artifacts.

    Args:
        params:        Hyperparameters dict (model_id, lora_r, lr, etc.)
        metrics:       Training metrics (train_loss, eval_loss, etc.)
        artifacts_dir: Optional path to folder of files to attach
    """
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        if artifacts_dir:
            mlflow.log_artifacts(artifacts_dir)
        run_id = mlflow.active_run().info.run_id
        print(f"[MLflow] Run logged. ID: {run_id}")
        return run_id


def log_eval_results(finetuned_score: float, baseline_score: float, improvement_pct: float):
    """
    Logs evaluation comparison metrics between fine-tuned model and baseline.

    Args:
        finetuned_score:  Average judge score for the fine-tuned model (0–1)
        baseline_score:   Average judge score for GPT-4o zero-shot baseline (0–1)
        improvement_pct:  Relative improvement percentage
    """
    with mlflow.start_run():
        mlflow.log_metrics({
            "finetuned_avg_score":  finetuned_score,
            "baseline_avg_score":   baseline_score,
            "improvement_pct":      improvement_pct,
        })
        print(f"[MLflow] Eval results logged. Improvement: +{improvement_pct}%")
