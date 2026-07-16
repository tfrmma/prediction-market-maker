"""
Secret loading. Kalshi's private key PEM was coming straight out of an
env var in plaintext, fine for local dev, not fine for anything that
looks like production.

This tries a secrets manager first (currently AWS Secrets Manager, since
that's what we run against, would take a `provider` arg to extend to
GCP/Vault later) and falls back to the env var with a loud warning if no
secret ARN is configured. boto3 is imported lazily so this module, and
anything that imports it, doesn't gain a hard dependency on it for
people who just want the env var path.
"""
from __future__ import annotations

import os
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)


def load_secret(env_var: str, secrets_manager_arn_env_var: Optional[str] = None) -> str:
    """
    Resolve a secret value. If `secrets_manager_arn_env_var` is set and
    the corresponding env var holds an ARN, fetch from AWS Secrets
    Manager. Otherwise fall back to reading `env_var` directly.
    """
    arn = os.environ.get(secrets_manager_arn_env_var, "") if secrets_manager_arn_env_var else ""

    if arn:
        return _load_from_aws_secrets_manager(arn)

    logger.warning(
        "secret_from_plain_env_var",
        env_var=env_var,
        hint=f"set {secrets_manager_arn_env_var} to source this from AWS Secrets Manager instead",
    )
    return os.environ[env_var]


def _load_from_aws_secrets_manager(secret_arn: str) -> str:
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError(
            "boto3 is required to load secrets from AWS Secrets Manager, "
            "pip install boto3 or unset the *_SECRET_ARN env var to fall back to plain env vars"
        ) from exc

    client = boto3.client("secretsmanager")
    resp = client.get_secret_value(SecretId=secret_arn)
    secret = resp.get("SecretString")
    if secret is None:
        raise RuntimeError(f"secret {secret_arn} has no SecretString (binary secrets aren't supported here)")

    logger.info("secret_loaded_from_aws", secret_arn=secret_arn)
    return secret
