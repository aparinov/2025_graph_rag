# -*- coding: utf-8 -*-
"""LLM client management."""

from typing import Dict, Tuple
from langchain_openai import ChatOpenAI
from app.config import OPENAI_API_KEY
from app.clients.http_client import get_http_client


# LLM cache
_LLM_CACHE: Dict[Tuple[str, float], ChatOpenAI] = {}


def get_llm(model_name: str, temperature: float = 0.0) -> ChatOpenAI:
    """Get cached LLM instance."""
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set")
    key = (model_name, temperature)
    if key not in _LLM_CACHE:
        _LLM_CACHE[key] = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            api_key=OPENAI_API_KEY,
            http_client=get_http_client(),
        )
    return _LLM_CACHE[key]
