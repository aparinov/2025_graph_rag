# -*- coding: utf-8 -*-
"""Entity extraction service."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

try:
    import json_repair
except ImportError:
    raise ImportError(
        "The 'json_repair' library is required. Please install it with 'pip install json-repair'"
    )

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from app.config import MAX_WORKERS, MAX_CHUNK_CHARS
from app.models.prompts import ENTITY_PROMPT
from app.utils.text_utils import strip_code_fences, sanitize_properties


class EntityService:
    """Service for extracting entities from text chunks."""

    def extract_entities_parallel(
        self, chunks: List[str], llm: ChatOpenAI
    ) -> List[List[Dict[str, Any]]]:
        """Extract entities from chunks in parallel."""
        chain = ENTITY_PROMPT | llm | StrOutputParser()
        results: List[List[Dict[str, Any]]] = [[] for _ in chunks]

        def _extract(text: str) -> List[Dict[str, Any]]:
            response = chain.invoke({"text": text[:MAX_CHUNK_CHARS]})
            parsed = json_repair.loads(strip_code_fences(response))
            if not isinstance(parsed, list):
                return []
            clean_entities = []
            for entity in parsed:
                if not isinstance(entity, dict):
                    continue
                name = str(entity.get("name", "")).strip()
                entity_type = str(entity.get("type", "")).strip()
                if not name or not entity_type:
                    continue
                properties = entity.get("properties")
                if not isinstance(properties, dict):
                    properties = {}
                properties = sanitize_properties(properties)
                clean_entities.append(
                    {"name": name, "type": entity_type, "properties": properties}
                )
            return clean_entities

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(_extract, text): idx for idx, text in enumerate(chunks)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    print(f"Ошибка извлечения сущностей (чанк {idx + 1}): {exc}")
                    results[idx] = []

        return results
