# -*- coding: utf-8 -*-
"""Relation extraction service."""

import json
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
from app.models.prompts import RELATION_PROMPT
from app.utils.text_utils import strip_code_fences, sanitize_properties


class RelationService:
    """Service for extracting relations between entities."""

    def extract_relations_parallel(
        self,
        chunks: List[str],
        entities_by_chunk: List[List[Dict[str, Any]]],
        llm: ChatOpenAI,
    ) -> List[Dict[str, Any]]:
        """Extract relations from chunks in parallel."""
        chain = RELATION_PROMPT | llm | StrOutputParser()
        relations: List[Dict[str, Any]] = []

        def _extract(text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            if len(entities) < 2:
                return []
            payload = [
                {"id": e["id"], "name": e["name"], "type": e["type"]} for e in entities
            ]
            response = chain.invoke(
                {
                    "text": text[:MAX_CHUNK_CHARS],
                    "entities": json.dumps(payload, ensure_ascii=False),
                }
            )
            parsed = json_repair.loads(strip_code_fences(response))
            if not isinstance(parsed, list):
                return []
            allowed_ids = {e["id"] for e in entities}
            clean_relations = []
            for rel in parsed:
                if not isinstance(rel, dict):
                    continue
                source_id = str(rel.get("source_id", "")).strip()
                target_id = str(rel.get("target_id", "")).strip()
                rel_type = str(rel.get("type", "")).strip()
                if not source_id or not target_id or not rel_type:
                    continue
                if source_id == target_id:
                    continue
                if source_id not in allowed_ids or target_id not in allowed_ids:
                    continue
                properties = rel.get("properties")
                if not isinstance(properties, dict):
                    properties = {}
                properties = sanitize_properties(properties)
                clean_relations.append(
                    {
                        "source_id": source_id,
                        "target_id": target_id,
                        "type": rel_type,
                        "properties": properties,
                    }
                )
            return clean_relations

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {}
            for text, entities in zip(chunks, entities_by_chunk):
                futures[executor.submit(_extract, text, entities)] = True
            for future in as_completed(futures):
                try:
                    relations.extend(future.result())
                except Exception as exc:
                    print(f"Ошибка извлечения отношений: {exc}")

        return relations
