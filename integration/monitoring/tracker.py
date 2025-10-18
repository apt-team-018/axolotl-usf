"""Multi-backend metrics tracker."""

import os
from typing import Dict, Any, Optional, List
from integration.monitoring.mongodb_logger import MongoDBLogger


class MetricsTracker:
    """Unified metrics tracking across multiple backends."""

    def __init__(self, config: Dict[str, Any], job_id: Optional[str] = None):
        """
        Initialize metrics tracker.

        Args:
            config: Monitoring configuration with backend settings
            job_id: Job ID for tracking
        """
        self.job_id = job_id or os.getenv('JOB_ID', 'unknown')
        self.backends = []
        self.backend_instances = {}

        # Initialize W&B if configured
        if config.get("wandb"):
            self._init_wandb(config["wandb"])

        # Initialize MLflow if configured
        if config.get("mlflow"):
            self._init_mlflow(config["mlflow"])

        # Initialize TensorBoard if configured
        if config.get("tensorboard"):
            self._init_tensorboard(config["tensorboard"])

        # Initialize MongoDB if configured
        if config.get("mongodb"):
            self._init_mongodb(config["mongodb"])

    def _init_wandb(self, config: Dict[str, Any]) -> None:
        """Initialize Weights & Biases."""
        try:
            import wandb

            wandb.init(
                project=config.get("project", "axolotl-training"),
                entity=config.get("entity"),
                name=config.get("run_name", self.job_id),
                config=config.get("config", {}),
                tags=config.get("tags", []),
                notes=config.get("notes"),
            )

            self.backends.append("wandb")
            self.backend_instances["wandb"] = wandb
            print("✅ W&B initialized")

        except ImportError:
            print("⚠️ wandb not installed, skipping W&B integration")
        except Exception as e:
            print(f"⚠️ Failed to initialize W&B: {e}")

    def _init_mlflow(self, config: Dict[str, Any]) -> None:
        """Initialize MLflow."""
        try:
            import mlflow

            tracking_uri = config.get("tracking_uri", "http://localhost:5000")
            mlflow.set_tracking_uri(tracking_uri)

            experiment_name = config.get("experiment_name", "axolotl-training")
            mlflow.set_experiment(experiment_name)

            mlflow.start_run(run_name=config.get("run_name", self.job_id))

            # Log parameters if provided
            if "params" in config:
                mlflow.log_params(config["params"])

            self.backends.append("mlflow")
            self.backend_instances["mlflow"] = mlflow
            print(f"✅ MLflow initialized (tracking URI: {tracking_uri})")

        except ImportError:
            print("⚠️ mlflow not installed, skipping MLflow integration")
        except Exception as e:
            print(f"⚠️ Failed to initialize MLflow: {e}")

    def _init_tensorboard(self, config: Dict[str, Any]) -> None:
        """Initialize TensorBoard."""
        try:
            from torch.utils.tensorboard import SummaryWriter

            log_dir = config.get("log_dir", f"./runs/{self.job_id}")
            writer = SummaryWriter(log_dir=log_dir)

            self.backends.append("tensorboard")
            self.backend_instances["tensorboard"] = writer
            print(f"✅ TensorBoard initialized (log dir: {log_dir})")

        except ImportError:
            print("⚠️ tensorboard not installed, skipping TensorBoard integration")
        except Exception as e:
            print(f"⚠️ Failed to initialize TensorBoard: {e}")

    def _init_mongodb(self, config: Dict[str, Any]) -> None:
        """Initialize MongoDB logger."""
        try:
            mongo_logger = MongoDBLogger(
                mongo_uri=config["uri"],
                database=config.get("database", "training"),
                collection=config.get("collection", "metrics"),
                job_id=self.job_id
            )

            self.backends.append("mongodb")
            self.backend_instances["mongodb"] = mongo_logger
            print("✅ MongoDB logger initialized")

        except Exception as e:
            print(f"⚠️ Failed to initialize MongoDB: {e}")

    def log_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        """
        Log metrics to all configured backends.

        Args:
            metrics: Dictionary of metrics
            step: Training step number
        """
        # W&B
        if "wandb" in self.backends:
            try:
                self.backend_instances["wandb"].log(metrics, step=step)
            except Exception as e:
                print(f"⚠️ Failed to log to W&B: {e}")

        # MLflow
        if "mlflow" in self.backends:
            try:
                mlflow = self.backend_instances["mlflow"]
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(key, value, step=step)
            except Exception as e:
                print(f"⚠️ Failed to log to MLflow: {e}")

        # TensorBoard
        if "tensorboard" in self.backends:
            try:
                writer = self.backend_instances["tensorboard"]
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        writer.add_scalar(key, value, step)
            except Exception as e:
                print(f"⚠️ Failed to log to TensorBoard: {e}")

        # MongoDB
        if "mongodb" in self.backends:
            try:
                mongo_logger = self.backend_instances["mongodb"]
                mongo_logger.log_training_step(metrics, step)
            except Exception as e:
                print(f"⚠️ Failed to log to MongoDB: {e}")

    def log_checkpoint(self, checkpoint_info: Dict[str, Any]) -> None:
        """
        Log checkpoint save event.

        Args:
            checkpoint_info: Checkpoint information
        """
        # MongoDB
        if "mongodb" in self.backends:
            try:
                mongo_logger = self.backend_instances["mongodb"]
                mongo_logger.log_checkpoint(checkpoint_info)
            except Exception as e:
                print(f"⚠️ Failed to log checkpoint to MongoDB: {e}")

    def log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Log custom event.

        Args:
            event_type: Type of event
            data: Event data
        """
        # MongoDB
        if "mongodb" in self.backends:
            try:
                mongo_logger = self.backend_instances["mongodb"]
                mongo_logger.log_event(event_type, data)
            except Exception as e:
                print(f"⚠️ Failed to log event to MongoDB: {e}")

    def finish(self) -> None:
        """Finish tracking and close all backends."""
        if "wandb" in self.backends:
            try:
                self.backend_instances["wandb"].finish()
            except Exception as e:
                print(f"⚠️ Failed to finish W&B: {e}")

        if "mlflow" in self.backends:
            try:
                self.backend_instances["mlflow"].end_run()
            except Exception as e:
                print(f"⚠️ Failed to end MLflow run: {e}")

        if "tensorboard" in self.backends:
            try:
                self.backend_instances["tensorboard"].close()
            except Exception as e:
                print(f"⚠️ Failed to close TensorBoard: {e}")

        if "mongodb" in self.backends:
            try:
                self.backend_instances["mongodb"].close()
            except Exception as e:
                print(f"⚠️ Failed to close MongoDB: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.finish()
