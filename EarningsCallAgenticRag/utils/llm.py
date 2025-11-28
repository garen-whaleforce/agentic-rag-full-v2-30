"""Shared LLM client helpers with Azure-first, OpenAI-fallback selection."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Tuple

from langchain_openai import OpenAIEmbeddings
from openai import AzureOpenAI, OpenAI

DEFAULT_AZURE_VERSION = "2024-12-01-preview"


def _azure_settings(creds: Dict[str, str]) -> Tuple[str | None, str | None, str, Dict[str, str], str | None]:
    """Collect Azure config from creds + env and return (key, endpoint, version, deployments, embedding_deployment)."""
    key = creds.get("azure_api_key") or os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = creds.get("azure_endpoint") or os.getenv("AZURE_OPENAI_ENDPOINT")
    version = creds.get("azure_api_version") or os.getenv("AZURE_OPENAI_API_VERSION") or DEFAULT_AZURE_VERSION
    deployments = creds.get("azure_deployments") or {}

    env_deployments = {}
    gpt5 = os.getenv("AZURE_OPENAI_DEPLOYMENT_GPT5")
    if gpt5:
        env_deployments["gpt-5-mini"] = gpt5
    gpt4o = os.getenv("AZURE_OPENAI_DEPLOYMENT_GPT4O")
    if gpt4o:
        env_deployments["gpt-4o-mini"] = gpt4o
    deployments = deployments or env_deployments

    embedding_dep = creds.get("azure_embedding_deployment") or os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

    return key, endpoint, version, deployments, embedding_dep


def build_chat_client(
    creds: Dict[str, str],
    requested_model: str,
    prefer_openai: bool = False,
) -> Tuple[OpenAI | AzureOpenAI, str]:
    """
    Return (client, model_name) where model_name is mapped to Azure deployment if available.
    Azure settings win; fallback to public OpenAI key.
    """
    if prefer_openai:
        api_key = creds.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("No OpenAI/Azure API key configured.")
        return OpenAI(api_key=api_key), requested_model

    azure_key, azure_endpoint, azure_version, deployments, _ = _azure_settings(creds)
    if azure_key and azure_endpoint:
        deployment = deployments.get(requested_model, requested_model)
        client = AzureOpenAI(
            api_key=azure_key,
            azure_endpoint=azure_endpoint,
            api_version=azure_version,
        )
        return client, deployment

    api_key = creds.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("No OpenAI/Azure API key configured.")
    return OpenAI(api_key=api_key), requested_model


def build_embeddings(creds: Dict[str, str], model: str = "text-embedding-3-small", prefer_openai: bool = False) -> OpenAIEmbeddings:
    """Return embeddings client; use Azure deployment if configured, else OpenAI."""
    if prefer_openai:
        api_key = creds.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("No OpenAI API key configured.")
        return OpenAIEmbeddings(openai_api_key=api_key, model=model)

    azure_key, azure_endpoint, azure_version, deployments, embedding_dep = _azure_settings(creds)
    if azure_key and azure_endpoint:
        deployment = embedding_dep or deployments.get(model)
        # If no explicit Azure embedding deployment is configured, fall back to OpenAI.
        if deployment:
            return OpenAIEmbeddings(
                model=deployment,
                deployment=deployment,
                openai_api_key=azure_key,
                azure_endpoint=azure_endpoint,
                openai_api_version=azure_version,
            )

    api_key = creds.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("No OpenAI API key configured.")
    return OpenAIEmbeddings(openai_api_key=api_key, model=model)


def load_credentials(path: str | Path) -> Dict[str, str]:
    """Load JSON credentials file."""
    return json.loads(Path(path).read_text())
