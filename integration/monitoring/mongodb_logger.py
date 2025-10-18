"""MongoDB logger for real-time training metrics."""

import os
from datetime import datetime
from typing import Dict, Any, Optional

try:
    from pymongo import MongoClient
    from pymongo.errors import PyMongoError
    HAS_PYMONGO = True
except ImportError:
    HAS_PYMONGO = False


class MongoDBLogger:
    """Real-time training metrics logger to MongoDB."""

    def __init__(
        self,
        mongo_uri: str,
        database: str,
        collection: str,
        job_id: Optional[str] = None
    ):
        """
        Initialize MongoDB logger.

        Args:
            mongo_uri: MongoDB connection URI
            database: Database name
            collection: Collection name
            job_id: Optional job ID for tracking
        """
        if not HAS_PYMONGO:
            raise ImportError(
                "pymongo is required for MongoDBLogger. "
                "Install with: pip install pymongo"
            )

        self.client = MongoClient(mongo_uri)
        self.db = self.client[database]
        self.collection = self.db[collection]
        self.job_id = job_id or os.getenv('JOB_ID', 'unknown')

        # Test connection
        try:
            self.client.server_info()
            print(f"✅ Connected to MongoDB: {database}.{collection}")
        except PyMongoError as e:
            print(f"⚠️ MongoDB connection warning: {e}")

    def log_training_step(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """
        Log metrics for a training step.

        Args:
            metrics: Dictionary of metrics to log
            step: Training step number (optional, extracted from metrics if not provided)
        """
        step = step or metrics.get("step", metrics.get("global_step", 0))

        doc = {
            "job_id": self.job_id,
            "timestamp": datetime.utcnow(),
            "event_type": "training_step",
            "step": step,
            "node_id": int(os.getenv("NODE_RANK", os.getenv("RANK", 0))),
            "gpu_id": int(os.getenv("LOCAL_RANK", 0)),
            **metrics
        }

        try:
            self.collection.insert_one(doc)
        except PyMongoError as e:
            print(f"⚠️ Failed to log to MongoDB: {e}")

    def log_checkpoint(self, checkpoint_info: Dict[str, Any]) -> None:
        """
        Log checkpoint save event.

        Args:
            checkpoint_info: Information about the checkpoint
        """
        doc = {
            "job_id": self.job_id,
            "timestamp": datetime.utcnow(),
            "event_type": "checkpoint",
            "node_id": int(os.getenv("NODE_RANK", os.getenv("RANK", 0))),
            **checkpoint_info
        }

        try:
            self.collection.insert_one(doc)
        except PyMongoError as e:
            print(f"⚠️ Failed to log checkpoint to MongoDB: {e}")

    def log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Log a custom event.

        Args:
            event_type: Type of event
            data: Event data
        """
        doc = {
            "job_id": self.job_id,
            "timestamp": datetime.utcnow(),
            "event_type": event_type,
            "node_id": int(os.getenv("NODE_RANK", os.getenv("RANK", 0))),
            **data
        }

        try:
            self.collection.insert_one(doc)
        except PyMongoError as e:
            print(f"⚠️ Failed to log event to MongoDB: {e}")

    def log_error(self, error: str, traceback: Optional[str] = None) -> None:
        """
        Log an error.

        Args:
            error: Error message
            traceback: Optional error traceback
        """
        doc = {
            "job_id": self.job_id,
            "timestamp": datetime.utcnow(),
            "event_type": "error",
            "error": error,
            "traceback": traceback,
            "node_id": int(os.getenv("NODE_RANK", os.getenv("RANK", 0))),
        }

        try:
            self.collection.insert_one(doc)
        except PyMongoError as e:
            print(f"⚠️ Failed to log error to MongoDB: {e}")

    def close(self) -> None:
        """Close MongoDB connection."""
        if self.client:
            self.client.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
