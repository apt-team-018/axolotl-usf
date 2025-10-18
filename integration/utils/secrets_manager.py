"""
Secrets Manager - Production-grade credential management.
Supports HashiCorp Vault, AWS Secrets Manager, and environment variables.
"""

import os
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class SecretsManager(ABC):
    """Abstract base class for secrets management"""

    @abstractmethod
    def get_secret(self, path: str) -> Dict[str, Any]:
        """Retrieve secret from secrets store"""
        pass

    @abstractmethod
    def get_secret_value(self, path: str, key: str) -> str:
        """Retrieve specific value from secret"""
        pass


class VaultSecretsManager(SecretsManager):
    """HashiCorp Vault secrets manager"""

    def __init__(self, vault_addr: str, vault_token: str, mount_point: str = "secret"):
        """
        Initialize Vault client.

        Args:
            vault_addr: Vault server address (e.g., https://vault.example.com:8200)
            vault_token: Vault authentication token
            mount_point: KV secrets engine mount point (default: secret)
        """
        try:
            import hvac
            self.client = hvac.Client(url=vault_addr, token=vault_token)
            self.mount_point = mount_point

            if not self.client.is_authenticated():
                raise ValueError("Vault authentication failed")

            logger.info(f"✅ Connected to Vault at {vault_addr}")
        except ImportError:
            raise ImportError("hvac library required. Install with: pip install hvac")

    def get_secret(self, path: str) -> Dict[str, Any]:
        """Get all values from secret path"""
        try:
            secret = self.client.secrets.kv.v2.read_secret_version(
                path=path,
                mount_point=self.mount_point
            )
            return secret['data']['data']
        except Exception as e:
            logger.error(f"Failed to retrieve secret from Vault: {path}")
            raise RuntimeError(f"Vault secret retrieval failed: {e}")

    def get_secret_value(self, path: str, key: str) -> str:
        """Get specific value from secret"""
        secret = self.get_secret(path)
        if key not in secret:
            raise KeyError(f"Key '{key}' not found in secret '{path}'")
        return secret[key]


class AWSSecretsManager(SecretsManager):
    """AWS Secrets Manager"""

    def __init__(self, region: str = 'us-east-1'):
        """
        Initialize AWS Secrets Manager client.

        Args:
            region: AWS region
        """
        try:
            import boto3
            self.client = boto3.client('secretsmanager', region_name=region)
            logger.info(f"✅ Connected to AWS Secrets Manager in {region}")
        except ImportError:
            raise ImportError("boto3 library required. Install with: pip install boto3")

    def get_secret(self, secret_id: str) -> Dict[str, Any]:
        """Get secret from AWS Secrets Manager"""
        try:
            response = self.client.get_secret_value(SecretId=secret_id)
            secret_string = response['SecretString']
            return json.loads(secret_string)
        except Exception as e:
            logger.error(f"Failed to retrieve secret from AWS: {secret_id}")
            raise RuntimeError(f"AWS Secrets Manager retrieval failed: {e}")

    def get_secret_value(self, secret_id: str, key: str) -> str:
        """Get specific value from secret"""
        secret = self.get_secret(secret_id)
        if key not in secret:
            raise KeyError(f"Key '{key}' not found in secret '{secret_id}'")
        return secret[key]


class EnvironmentSecretsManager(SecretsManager):
    """
    Environment variable-based secrets (for development/testing only).
    NOT RECOMMENDED FOR PRODUCTION.
    """

    def __init__(self):
        logger.warning("⚠️  Using environment variables for secrets - NOT RECOMMENDED FOR PRODUCTION")

    def get_secret(self, prefix: str) -> Dict[str, Any]:
        """Get all env vars with given prefix"""
        secrets = {}
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Remove prefix from key
                clean_key = key[len(prefix):].lower()
                secrets[clean_key] = value
        return secrets

    def get_secret_value(self, prefix: str, key: str) -> str:
        """Get specific env var"""
        env_key = f"{prefix}{key.upper()}"
        value = os.getenv(env_key)
        if value is None:
            raise KeyError(f"Environment variable '{env_key}' not found")
        return value


def get_secrets_manager(
    manager_type: Optional[str] = None,
    **kwargs
) -> SecretsManager:
    """
    Factory function to create appropriate secrets manager.

    Args:
        manager_type: Type of secrets manager ('vault', 'aws', 'env')
                     If None, auto-detect from environment
        **kwargs: Manager-specific configuration

    Returns:
        SecretsManager instance

    Examples:
        >>> # Vault
        >>> mgr = get_secrets_manager(
        ...     'vault',
        ...     vault_addr='https://vault.example.com:8200',
        ...     vault_token=os.getenv('VAULT_TOKEN')
        ... )

        >>> # AWS
        >>> mgr = get_secrets_manager('aws', region='us-east-1')

        >>> # Environment (dev only)
        >>> mgr = get_secrets_manager('env')
    """
    # Auto-detect if not specified
    if manager_type is None:
        if os.getenv('VAULT_ADDR'):
            manager_type = 'vault'
        elif os.getenv('AWS_REGION') or os.getenv('AWS_DEFAULT_REGION'):
            manager_type = 'aws'
        else:
            manager_type = 'env'
            logger.warning("No secrets manager configured, falling back to environment variables")

    manager_type = manager_type.lower()

    if manager_type == 'vault':
        vault_addr = kwargs.get('vault_addr') or os.getenv('VAULT_ADDR')
        vault_token = kwargs.get('vault_token') or os.getenv('VAULT_TOKEN')

        if not vault_addr or not vault_token:
            raise ValueError("Vault requires VAULT_ADDR and VAULT_TOKEN")

        return VaultSecretsManager(
            vault_addr=vault_addr,
            vault_token=vault_token,
            mount_point=kwargs.get('mount_point', 'secret')
        )

    elif manager_type == 'aws':
        return AWSSecretsManager(
            region=kwargs.get('region', os.getenv('AWS_REGION', 'us-east-1'))
        )

    elif manager_type == 'env':
        return EnvironmentSecretsManager()

    else:
        raise ValueError(f"Unknown secrets manager type: {manager_type}")


# Global secrets manager instance (lazy initialization)
_secrets_manager: Optional[SecretsManager] = None


def init_secrets_manager(manager_type: Optional[str] = None, **kwargs):
    """Initialize global secrets manager"""
    global _secrets_manager
    _secrets_manager = get_secrets_manager(manager_type, **kwargs)
    logger.info(f"✅ Secrets manager initialized: {type(_secrets_manager).__name__}")


def get_secret(path: str) -> Dict[str, Any]:
    """Get secret using global secrets manager"""
    if _secrets_manager is None:
        init_secrets_manager()
    return _secrets_manager.get_secret(path)


def get_secret_value(path: str, key: str) -> str:
    """Get specific secret value using global secrets manager"""
    if _secrets_manager is None:
        init_secrets_manager()
    return _secrets_manager.get_secret_value(path, key)
