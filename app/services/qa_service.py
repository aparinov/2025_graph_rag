# -*- coding: utf-8 -*-
"""Q&A service for answering questions based on documents."""

import re
from typing import List

from langchain_core.output_parsers import StrOutputParser

from app.config import OPENAI_QA_MODEL
from app.clients.llm_client import get_llm
from app.models.prompts import QA_PROMPT, TRANSLATE_PROMPT


class QAService:
    """Service for question answering."""

    def __init__(self, qdrant_manager):
        """Initialize QA service."""
        self.qdrant = qdrant_manager

    def _detect_language(self, text: str) -> str:
        """Detect if text is primarily Russian or English.

        Returns:
            "ru" or "en"
        """
        cyrillic_count = len(re.findall(r'[а-яА-ЯёЁ]', text))
        latin_count = len(re.findall(r'[a-zA-Z]', text))
        return "ru" if cyrillic_count >= latin_count else "en"

    def _translate(self, text: str, target_language: str) -> str:
        """Translate text to target language using LLM.

        Args:
            text: Text to translate
            target_language: "Russian" or "English"

        Returns:
            Translated text
        """
        llm = get_llm(OPENAI_QA_MODEL, temperature=0.1)
        chain = TRANSLATE_PROMPT | llm | StrOutputParser()
        return chain.invoke({"text": text, "target_language": target_language})

    def answer_question(self, question: str, collection_names: List[str]) -> str:
        """Answer question using multi-collection search with bilingual support.

        Detects query language, translates for cross-language search,
        and returns answer in both Russian and English.

        Args:
            question: User question
            collection_names: List of collections to search

        Returns:
            Bilingual answer text
        """
        lang = self._detect_language(question)
        print(f"[DEBUG] Question: {question}")
        print(f"[DEBUG] Detected language: {lang}")
        print(f"[DEBUG] Searching collections: {collection_names}")

        # Translate query for cross-language search
        if lang == "ru":
            translated_query = self._translate(question, "English")
        else:
            translated_query = self._translate(question, "Russian")

        print(f"[DEBUG] Translated query: {translated_query}")

        # Search with original query
        original_results = self.qdrant.search_multi_collection(
            question, collection_names, k=5
        )
        # Search with translated query
        translated_results = self.qdrant.search_multi_collection(
            translated_query, collection_names, k=5
        )

        # Deduplicate by (collection_name, document_name, chunk_id)
        seen = set()
        search_results = []
        for r in original_results + translated_results:
            key = (r["collection_name"], r["document_name"], r["chunk_id"])
            if key not in seen:
                seen.add(key)
                search_results.append(r)

        # Sort by score descending
        search_results.sort(key=lambda x: x["score"], reverse=True)

        print(f"[DEBUG] Found {len(search_results)} relevant chunks (deduplicated)")

        if not search_results:
            return (
                "## RU\nВ текущих документах нет данных для ответа.\n\n"
                "## EN\nNo relevant data found in the current documents."
            )

        # Generate bilingual answer using LLM
        qa_llm = get_llm(OPENAI_QA_MODEL, temperature=0.2)
        chain = QA_PROMPT | qa_llm | StrOutputParser()

        # Format context from chunks with collection citations
        context = "\n\n".join(
            [
                f"[Collection: {r['collection_name']}, Document: {r['document_name']}, Chunk {r['chunk_id']}]\n{r['text']}"
                for r in search_results
            ]
        )

        answer = chain.invoke({"question": question, "results": context})

        print(f"[DEBUG] Generated bilingual answer")

        return answer
