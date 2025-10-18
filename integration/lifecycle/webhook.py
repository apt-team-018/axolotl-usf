"""
Webhook manager - PRODUCTION READY with security.
Handles deletion webhooks with retry, delayed execution, and SSRF prevention.

Security Features:
- SSRF (Server-Side Request Forgery) prevention
- URL validation and whitelisting
- Timeout enforcement
- Exponential backoff retry
"""

import os
import requests
import time
import threading
import ipaddress
import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from urllib.parse import urlparse

from integration.utils.retry import retry_on_exception

logger = logging.getLogger(__name__)


class WebhookManager:
    """
    Manages webhook calls with production-grade security.

    Security Features:
    - SSRF prevention (blocks private IPs)
    - URL validation
    - Domain whitelisting
    - Request timeout enforcement
    - Exponential backoff retry
    """

    def __init__(
        self,
        webhook_url: str,
        method: str = "POST",
        headers: Optional[Dict[str, str]] = None,
        body_template: Optional[str] = None,
        retry_attempts: int = 5,
        retry_delay_seconds: int = 30,
        allowed_domains: Optional[list] = None
    ):
        """
        Initialize webhook manager with security validation.

        Args:
            webhook_url: Webhook URL to call
            method: HTTP method (POST, DELETE, PUT)
            headers: HTTP headers (including auth)
            body_template: Request body template with variables
            retry_attempts: Number of retry attempts
            retry_delay_seconds: Delay between retries
            allowed_domains: Optional whitelist of allowed domains

        Raises:
            ValueError: If webhook URL fails security validation
        """
        # SECURITY: Validate webhook URL before storing
        self._validate_webhook_url(webhook_url, allowed_domains)

        self.webhook_url = webhook_url
        self.method = method.upper()
        self.headers = headers or {}
        self.body_template = body_template or "{}"
        self.retry_attempts = retry_attempts
        self.retry_delay_seconds = retry_delay_seconds
        self.allowed_domains = allowed_domains or []

        logger.info(f"✅ Webhook manager initialized: {method} {webhook_url}")

    def _validate_webhook_url(self, url: str, allowed_domains: Optional[list] = None):
        """
        Validate webhook URL to prevent SSRF attacks.

        Security Checks:
        - Valid HTTP/HTTPS scheme
        - Not targeting private/loopback IPs
        - Domain whitelist (if provided)

        Args:
            url: URL to validate
            allowed_domains: Optional list of allowed domains

        Raises:
            ValueError: If URL fails validation
        """
        try:
            parsed = urlparse(url)

            # Check 1: Only allow http/https
            if parsed.scheme not in ['http', 'https']:
                raise ValueError(
                    f"Invalid URL scheme: {parsed.scheme}. "
                    f"Only 'http' and 'https' are allowed."
                )

            # Check 2: URL must have a hostname
            if not parsed.hostname:
                raise ValueError("URL must have a valid hostname")

            # Check 3: Prevent access to private/loopback IPs (SSRF prevention)
            try:
                ip = ipaddress.ip_address(parsed.hostname)
                if ip.is_private or ip.is_loopback or ip.is_reserved:
                    raise ValueError(
                        f"Webhook URL cannot target private/loopback IP: {parsed.hostname}. "
                        f"This is blocked for security (SSRF prevention)."
                    )
            except ValueError as e:
                # Not an IP address, it's a hostname - this is OK
                if "does not appear to be" not in str(e):
                    raise

            # Check 4: Domain whitelist (if provided)
            if allowed_domains:
                if parsed.hostname not in allowed_domains:
                    raise ValueError(
                        f"Webhook domain not in whitelist: {parsed.hostname}. "
                        f"Allowed domains: {', '.join(allowed_domains)}"
                    )

            # Check 5: Environment variable whitelist (if set)
            env_whitelist = os.getenv('ALLOWED_WEBHOOK_DOMAINS', '').split(',')
            env_whitelist = [d.strip() for d in env_whitelist if d.strip()]

            if env_whitelist and parsed.hostname not in env_whitelist:
                logger.warning(
                    f"Webhook domain {parsed.hostname} not in environment whitelist. "
                    f"Consider adding to ALLOWED_WEBHOOK_DOMAINS."
                )

            logger.debug(f"Webhook URL validated: {url}")

        except Exception as e:
            logger.error(f"Webhook URL validation failed: {e}")
            raise ValueError(f"Invalid webhook URL: {e}")

    def _substitute_variables(self, template: str, variables: Dict[str, Any]) -> str:
        """
        Substitute variables in template string.

        Security: Uses simple string replacement (no code execution).
        """
        result = template
        for key, value in variables.items():
            placeholder = f"${{{key}}}"
            # Simple string replacement - no eval() or exec()
            result = result.replace(placeholder, str(value))
        return result

    def _call_webhook_with_retry(
        self,
        variables: Dict[str, Any],
        timeout: Tuple[int, int] = (5, 30)
    ) -> Tuple[bool, Optional[int], Optional[str]]:
        """
        Call webhook with exponential backoff retry.

        Args:
            variables: Variables to substitute
            timeout: (connect_timeout, read_timeout) in seconds

        Returns:
            Tuple of (success, status_code, response_text)
        """
        for attempt in range(self.retry_attempts):
            try:
                # Substitute variables in body
                body = self._substitute_variables(self.body_template, variables)

                # Substitute variables in headers
                headers = {
                    key: self._substitute_variables(value, variables)
                    for key, value in self.headers.items()
                }

                # Substitute variables in URL
                url = self._substitute_variables(self.webhook_url, variables)

                # SECURITY: Re-validate URL after variable substitution
                self._validate_webhook_url(url, self.allowed_domains)

                logger.info(f"Calling webhook (attempt {attempt + 1}/{self.retry_attempts}): {self.method} {url}")
                logger.debug(f"Request body: {body[:200]}...")  # Log first 200 chars only

                # Make request with timeout
                response = requests.request(
                    method=self.method,
                    url=url,
                    headers=headers,
                    data=body if self.method in ['POST', 'PUT', 'PATCH'] else None,
                    timeout=timeout,  # (connect timeout, read timeout)
                    allow_redirects=False  # Security: Don't follow redirects
                )

                # Check response
                response.raise_for_status()

                logger.info(
                    f"✅ Webhook call successful: {response.status_code}",
                    extra={"status_code": response.status_code, "attempt": attempt + 1}
                )

                return True, response.status_code, response.text

            except requests.Timeout as e:
                logger.warning(f"Webhook timeout (attempt {attempt + 1}): {e}")
                if attempt < self.retry_attempts - 1:
                    wait_time = min(2 ** attempt * self.retry_delay_seconds, 300)  # Max 5 min
                    logger.info(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Webhook failed after {self.retry_attempts} attempts (timeout)")
                    raise

            except requests.RequestException as e:
                logger.warning(f"Webhook request failed (attempt {attempt + 1}): {e}")
                if attempt < self.retry_attempts - 1:
                    # Exponential backoff
                    wait_time = min(2 ** attempt * self.retry_delay_seconds, 300)
                    logger.info(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Webhook failed after {self.retry_attempts} attempts")
                    raise

        return False, None, None

    def call_webhook(
        self,
        variables: Dict[str, Any],
        timeout: Tuple[int, int] = (5, 30)
    ) -> Tuple[bool, Optional[int], Optional[str]]:
        """
        Call webhook with retry logic (wrapper for backward compatibility).

        Args:
            variables: Variables to substitute in body template and headers
            timeout: (connect_timeout, read_timeout) tuple in seconds

        Returns:
            Tuple of (success, status_code, response_text)
        """
        return self._call_webhook_with_retry(variables, timeout)

    def call_webhook_delayed(
        self,
        delay_seconds: int,
        variables: Dict[str, Any],
        callback: Optional[callable] = None
    ) -> threading.Thread:
        """
        Call webhook after a delay (non-blocking).

        Args:
            delay_seconds: Delay before calling webhook
            variables: Variables for webhook
            callback: Optional callback after webhook completes

        Returns:
            Thread object (already started)
        """
        def delayed_call():
            try:
                logger.info(f"Waiting {delay_seconds}s before webhook call...")
                time.sleep(delay_seconds)

                success, status_code, response = self.call_webhook(variables)

                if callback:
                    callback(success, status_code, response)

            except Exception as e:
                logger.error(f"Delayed webhook call failed: {e}")
                if callback:
                    callback(False, None, str(e))

        thread = threading.Thread(
            target=delayed_call,
            daemon=False,
            name=f"webhook-delayed-{int(time.time())}"
        )
        thread.start()

        logger.info(f"Webhook scheduled to run in {delay_seconds}s")

        return thread

    def call_deletion_webhook(
        self,
        job_id: str,
        reason: str,
        status: str,
        pod_id: Optional[str] = None,
        delay_seconds: int = 0
    ) -> bool:
        """
        Call server deletion webhook.

        Args:
            job_id: Job identifier
            reason: Reason for deletion
            status: Final job status
            pod_id: Pod/server ID to delete
            delay_seconds: Delay before deletion

        Returns:
            True if webhook called successfully (or scheduled)
        """
        variables = {
            "JOB_ID": job_id,
            "REASON": reason,
            "STATUS": status,
            "POD_ID": pod_id or os.getenv("POD_ID", "unknown"),
            "DELETION_REASON": reason,
            "TIMESTAMP": datetime.utcnow().isoformat()
        }

        if delay_seconds > 0:
            # Delayed webhook
            logger.info(
                f"Scheduling deletion webhook in {delay_seconds}s "
                f"(job={job_id}, status={status})"
            )
            self.call_webhook_delayed(delay_seconds, variables)
            return True
        else:
            # Immediate webhook
            try:
                success, status_code, response = self.call_webhook(variables)
                return success
            except Exception as e:
                logger.error(f"Immediate deletion webhook failed: {e}")
                return False
