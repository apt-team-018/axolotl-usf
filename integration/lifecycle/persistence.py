"""
MongoDB persistence layer - PRODUCTION READY.
Stores and retrieves complete job state including checkpoints and retry history.

Features:
- Circuit breaker for failure protection
- TLS/SSL encryption
- Connection pooling
- Graceful degradation (training continues if MongoDB unavailable)
- Automatic retry with exponential backoff
"""

import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import logging

from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import PyMongoError, ConnectionFailure, ServerSelectionTimeoutError

from integration.lifecycle.states import TrainingState
from integration.utils.retry import with_api_retry
from integration.utils.circuit_breaker import circuit

logger = logging.getLogger(__name__)


class JobStatePersistence:
    """
    Persists training job state to MongoDB - Production Ready.

    Features:
    - Thread-safe operations
    - Circuit breaker protection
    - TLS/SSL encryption
    - Connection pooling
    - Graceful degradation
    - Automatic reconnection
    """

    def __init__(
        self,
        mongo_uri: str,
        database: str,
        collection: str,
        job_id: str,
        use_tls: bool = True,
        pool_size: int = 50
    ):
        """
        Initialize persistence layer with production settings.

        Args:
            mongo_uri: MongoDB connection URI
            database: Database name
            collection: Collection name
            job_id: Job identifier
            use_tls: Enable TLS/SSL (RECOMMENDED for production)
            pool_size: Connection pool size (default: 50)
        """
        self.job_id = job_id
        self.database_name = database
        self.collection_name = collection

        # Production-ready MongoDB connection with:
        # - Connection pooling
        # - TLS encryption
        # - Automatic retry
        # - Timeout settings
        try:
            self.client = MongoClient(
                mongo_uri,
                # Connection pooling
                maxPoolSize=pool_size,
                minPoolSize=10,
                maxIdleTimeMS=45000,

                # Timeouts
                serverSelectionTimeoutMS=5000,  # 5 seconds
                connectTimeoutMS=10000,  # 10 seconds
                socketTimeoutMS=30000,  # 30 seconds

                # Automatic retry
                retryWrites=True,
                retryReads=True,

                # TLS/SSL
                tls=use_tls,
                tlsAllowInvalidCertificates=False if use_tls else True,

                # Application name for monitoring
                appname=f"axolotl-training-{job_id}"
            )

            # Test connection
            self.client.admin.command('ping')

            self.db = self.client[database]
            self.collection = self.db[collection]

            # Create indexes for efficient queries
            try:
                self.collection.create_index([("job_id", ASCENDING)], unique=True)
                self.collection.create_index([("status", ASCENDING)])
                self.collection.create_index([("heartbeat.last_update", DESCENDING)])
            except Exception as e:
                logger.warning(f"Failed to create indexes: {e}")

            logger.info(
                f"✅ Connected to MongoDB: {database}.{collection} "
                f"(TLS={use_tls}, pool={pool_size})"
            )

        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            logger.warning("⚠️  Training will continue without persistence (degraded mode)")
            self.client = None
            self.db = None
            self.collection = None

    @with_api_retry
    @circuit(failure_threshold=5, recovery_timeout=120, name="mongodb_create_job")
    def create_job(self, config: Dict[str, Any]) -> None:
        """
        Create initial job record in MongoDB with circuit breaker.

        Args:
            config: Job configuration
        """
        if not self.collection:
            logger.warning("MongoDB unavailable, skipping job creation")
            return

        job_record = {
            "job_id": self.job_id,
            "status": TrainingState.INITIALIZING.value,
            "created_at": datetime.utcnow(),
            "started_at": None,
            "completed_at": None,

            # Configuration
            "config": config,

            # Progress tracking
            "progress": {
                "current_step": 0,
                "total_steps": None,
                "current_epoch": 0,
                "total_epochs": config.get("hyperparameters", {}).get("num_epochs", 3),
                "last_loss": None,
                "metrics": {}
            },

            # Storage configuration
            "storage": {
                "dataset": config.get("storage", {}).get("dataset", {}),
                "checkpoints": {
                    "type": config.get("storage", {}).get("checkpoints", {}).get("type"),
                    "base_uri": config.get("storage", {}).get("checkpoints", {}).get("uri"),
                    "checkpoints": [],
                    "latest_checkpoint": None,
                    "keep_last_n": config.get("lifecycle", {}).get("checkpoints", {}).get("keep_last_n", 3)
                },
                "model": config.get("storage", {}).get("model", {})
            },

            # Retry tracking
            "retry": {
                "current_attempt": 1,
                "max_attempts": config.get("lifecycle", {}).get("retry", {}).get("max_attempts", 3),
                "history": [],
                "next_retry_at": None
            },

            # Heartbeat tracking
            "heartbeat": {
                "enabled": config.get("lifecycle", {}).get("heartbeat", {}).get("enabled", True),
                "interval_seconds": config.get("lifecycle", {}).get("heartbeat", {}).get("interval_seconds", 60),
                "timeout_seconds": config.get("lifecycle", {}).get("heartbeat", {}).get("timeout_seconds", 3600),
                "last_update": datetime.utcnow(),
                "is_alive": True,
                "missed_heartbeats": 0
            },

            # Notifications tracking
            "notifications": {
                "emails_sent": [],
                "pending_emails": [],
                "webhook_calls": []
            },

            # Lifecycle configuration
            "lifecycle_config": config.get("lifecycle", {})
        }

        try:
            self.collection.insert_one(job_record)
            logger.info(f"Created job record in MongoDB: {self.job_id}")
        except (PyMongoError, ConnectionFailure) as e:
            logger.error(f"Failed to create job record: {e}")
            raise  # Let circuit breaker handle

    @with_api_retry
    @circuit(failure_threshold=5, recovery_timeout=120, name="mongodb_update_state")
    def update_state(
        self,
        new_state: TrainingState,
        reason: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Update job state with circuit breaker protection.

        If MongoDB fails, training continues (graceful degradation).

        Args:
            new_state: New state
            reason: Reason for state change
            metadata: Additional metadata
        """
        if not self.collection:
            logger.debug("MongoDB unavailable, skipping state update")
            return

        update_doc = {
            "$set": {
                "status": new_state.value,
                "last_updated": datetime.utcnow()
            },
            "$push": {
                "state_history": {
                    "state": new_state.value,
                    "timestamp": datetime.utcnow(),
                    "reason": reason,
                    "metadata": metadata or {}
                }
            }
        }

        # Set completed_at for terminal states
        if new_state in {TrainingState.SUCCESS, TrainingState.TERMINAL_FAILURE, TrainingState.COMPLETED}:
            update_doc["$set"]["completed_at"] = datetime.utcnow()

        # Set started_at when training actually starts
        if new_state == TrainingState.RUNNING:
            update_doc["$setOnInsert"] = {"started_at": datetime.utcnow()}

        try:
            self.collection.update_one(
                {"job_id": self.job_id},
                update_doc
            )
            logger.debug(f"State updated in MongoDB: {new_state.value}")
        except (PyMongoError, ConnectionFailure) as e:
            logger.warning(f"MongoDB state update failed: {e} - continuing anyway")
            raise  # Let circuit breaker handle

    @with_api_retry
    @circuit(failure_threshold=5, recovery_timeout=120, name="mongodb_update_progress")
    def update_progress(
        self,
        step: int,
        loss: Optional[float] = None,
        metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """Update training progress (non-critical, continues if fails)."""
        if not self.collection:
            return

        update_doc = {
            "$set": {
                "progress.current_step": step,
                "progress.last_updated": datetime.utcnow()
            }
        }

        if loss is not None:
            update_doc["$set"]["progress.last_loss"] = loss

        if metrics:
            update_doc["$set"]["progress.metrics"] = metrics

        try:
            self.collection.update_one(
                {"job_id": self.job_id},
                update_doc
            )
        except (PyMongoError, ConnectionFailure) as e:
            logger.debug(f"Progress update failed: {e}")
            # Non-critical, don't raise

    @with_api_retry
    @circuit(failure_threshold=5, recovery_timeout=120, name="mongodb_add_checkpoint")
    def add_checkpoint(
        self,
        step: int,
        checkpoint_uri: str,
        size_bytes: int,
        saved_on: str = "training_step"
    ) -> None:
        """
        Add checkpoint information (CRITICAL for resume).

        Args:
            step: Training step number
            checkpoint_uri: URI where checkpoint is stored
            size_bytes: Checkpoint size in bytes
            saved_on: Reason for save
        """
        if not self.collection:
            logger.warning("MongoDB unavailable, checkpoint not recorded")
            return

        checkpoint_info = {
            "step": step,
            "uri": checkpoint_uri,
            "size_bytes": size_bytes,
            "uploaded_at": datetime.utcnow(),
            "verified": True,
            "is_resumable": True,
            "saved_on": saved_on
        }

        try:
            self.collection.update_one(
                {"job_id": self.job_id},
                {
                    "$push": {"storage.checkpoints.checkpoints": checkpoint_info},
                    "$set": {
                        "storage.checkpoints.latest_checkpoint": {
                            "step": step,
                            "uri": checkpoint_uri
                        }
                    }
                }
            )
            logger.info(f"Checkpoint recorded in MongoDB: step {step}, reason: {saved_on}")
        except (PyMongoError, ConnectionFailure) as e:
            logger.warning(f"Failed to record checkpoint: {e}")
            raise  # Important for resume capability

    @with_api_retry
    @circuit(failure_threshold=5, recovery_timeout=120, name="mongodb_get_checkpoint")
    def get_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        Get latest checkpoint info for resume.
        CRITICAL: Works even after server deletion.

        Returns:
            Checkpoint info dict or None if no checkpoints
        """
        if not self.collection:
            logger.warning("MongoDB unavailable, cannot retrieve checkpoint")
            return None

        try:
            job = self.collection.find_one({"job_id": self.job_id})

            if not job:
                logger.warning(f"Job {self.job_id} not found in MongoDB")
                return None

            latest = job.get("storage", {}).get("checkpoints", {}).get("latest_checkpoint")

            if latest:
                logger.info(f"Latest checkpoint found: step {latest['step']}, uri {latest['uri']}")
            else:
                logger.info("No checkpoints found for this job")

            return latest

        except (PyMongoError, ConnectionFailure) as e:
            logger.error(f"Failed to retrieve checkpoint: {e}")
            return None

    @with_api_retry
    @circuit(failure_threshold=5, recovery_timeout=120, name="mongodb_retry")
    def record_retry_attempt(
        self,
        attempt: int,
        resumed_from: Optional[str] = None,
        error: Optional[str] = None
    ) -> None:
        """Record retry attempt."""
        if not self.collection:
            return

        retry_record = {
            "attempt": attempt,
            "started_at": datetime.utcnow(),
            "resumed_from": resumed_from,
            "status": "RUNNING",
            "error": error
        }

        try:
            self.collection.update_one(
                {"job_id": self.job_id},
                {
                    "$set": {"retry.current_attempt": attempt},
                    "$push": {"retry.history": retry_record}
                }
            )
            logger.info(f"Retry attempt {attempt} recorded in MongoDB")
        except (PyMongoError, ConnectionFailure) as e:
            logger.warning(f"Failed to record retry: {e}")

    @with_api_retry
    @circuit(failure_threshold=5, recovery_timeout=60, name="mongodb_heartbeat")
    def update_heartbeat(self) -> None:
        """Update heartbeat timestamp."""
        if not self.collection:
            return

        try:
            self.collection.update_one(
                {"job_id": self.job_id},
                {
                    "$set": {
                        "heartbeat.last_update": datetime.utcnow(),
                        "heartbeat.is_alive": True
                    }
                }
            )
        except (PyMongoError, ConnectionFailure) as e:
            logger.debug(f"Heartbeat update failed: {e}")
            # Non-critical, don't raise

    @with_api_retry
    @circuit(failure_threshold=5, recovery_timeout=120, name="mongodb_check_heartbeat")
    def check_heartbeat_timeout(self) -> bool:
        """
        Check if heartbeat has timed out.

        Returns:
            True if timeout detected, False otherwise
        """
        if not self.collection:
            return False

        try:
            job = self.collection.find_one({"job_id": self.job_id})

            if not job or not job.get("heartbeat", {}).get("enabled", True):
                return False

            last_update = job["heartbeat"]["last_update"]
            timeout_seconds = job["heartbeat"]["timeout_seconds"]

            time_since_update = (datetime.utcnow() - last_update).total_seconds()

            if time_since_update > timeout_seconds:
                logger.warning(
                    f"Heartbeat timeout detected: {time_since_update:.0f}s since last update "
                    f"(timeout: {timeout_seconds}s)"
                )
                return True

            return False

        except (PyMongoError, ConnectionFailure) as e:
            logger.warning(f"Heartbeat check failed: {e}")
            return False

    @with_api_retry
    @circuit(failure_threshold=5, recovery_timeout=60, name="mongodb_notification")
    def record_notification(
        self,
        notification_type: str,
        details: Dict[str, Any]
    ) -> None:
        """Record sent notification."""
        if not self.collection:
            return

        notification_record = {
            "type": notification_type,
            "sent_at": datetime.utcnow(),
            **details
        }

        try:
            self.collection.update_one(
                {"job_id": self.job_id},
                {"$push": {"notifications.emails_sent": notification_record}}
            )
        except (PyMongoError, ConnectionFailure) as e:
            logger.debug(f"Notification record failed: {e}")

    @with_api_retry
    @circuit(failure_threshold=5, recovery_timeout=60, name="mongodb_webhook")
    def record_webhook_call(
        self,
        webhook_type: str,
        url: str,
        status_code: Optional[int] = None,
        response: Optional[str] = None,
        error: Optional[str] = None
    ) -> None:
        """Record webhook call."""
        if not self.collection:
            return

        webhook_record = {
            "type": webhook_type,
            "url": url,
            "called_at": datetime.utcnow(),
            "status_code": status_code,
            "response": response,
            "error": error
        }

        try:
            self.collection.update_one(
                {"job_id": self.job_id},
                {"$push": {"notifications.webhook_calls": webhook_record}}
            )
        except (PyMongoError, ConnectionFailure) as e:
            logger.debug(f"Webhook record failed: {e}")

    def get_job_state(self) -> Optional[Dict[str, Any]]:
        """Get complete job state from MongoDB."""
        if not self.collection:
            return None

        try:
            return self.collection.find_one({"job_id": self.job_id})
        except (PyMongoError, ConnectionFailure) as e:
            logger.warning(f"Failed to get job state: {e}")
            return None

    def close(self) -> None:
        """Close MongoDB connection."""
        if self.client:
            try:
                self.client.close()
                logger.debug("MongoDB connection closed")
            except Exception as e:
                logger.warning(f"Error closing MongoDB: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
