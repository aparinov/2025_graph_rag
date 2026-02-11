# -*- coding: utf-8 -*-
"""Q&A service for answering questions based on documents."""

import re
from typing import Dict, List, Any

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

    def _search(self, question: str, collection_names: List[str]) -> List[Dict[str, Any]]:
        """Search across collections with bilingual queries and deduplication."""
        lang = self._detect_language(question)
        print(f"[DEBUG] Question: {question}")
        print(f"[DEBUG] Detected language: {lang}")
        print(f"[DEBUG] Searching collections: {collection_names}")

        if lang == "ru":
            translated_query = self._translate(question, "English")
        else:
            translated_query = self._translate(question, "Russian")

        print(f"[DEBUG] Translated query: {translated_query}")

        original_results = self.qdrant.search_multi_collection(
            question, collection_names, k=5
        )
        translated_results = self.qdrant.search_multi_collection(
            translated_query, collection_names, k=5
        )

        seen = set()
        search_results = []
        for r in original_results + translated_results:
            key = (r["collection_name"], r["document_name"], r["chunk_id"])
            if key not in seen:
                seen.add(key)
                search_results.append(r)

        search_results.sort(key=lambda x: x["score"], reverse=True)
        print(f"[DEBUG] Found {len(search_results)} relevant chunks (deduplicated)")
        return search_results

    def answer_question(self, question: str, collection_names: List[str]) -> str:
        """Answer question (legacy interface, returns only text)."""
        result = self.answer_question_with_sources(question, collection_names)
        return result["answer"]

    def answer_question_with_sources(
        self, question: str, collection_names: List[str]
    ) -> Dict[str, Any]:
        """Answer question and return both the answer and source documents.

        Returns:
            Dict with "answer" (str) and "sources" (list of source dicts)
        """
        search_results = self._search(question, collection_names)

        if not search_results:
            return {
                "answer": (
                    "## RU\nВ текущих документах нет данных для ответа.\n\n"
                    "## EN\nNo relevant data found in the current documents."
                ),
                "sources": [],
            }

        qa_llm = get_llm(OPENAI_QA_MODEL, temperature=0.2)
        chain = QA_PROMPT | qa_llm | StrOutputParser()

        context = "\n\n".join(
            [
                f"[Collection: {r['collection_name']}, Document: {r['document_name']}, Chunk {r['chunk_id']}]\n{r['text']}"
                for r in search_results
            ]
        )

        answer = chain.invoke({"question": question, "results": context})
        print(f"[DEBUG] Generated bilingual answer")

        # Deduplicate sources by document name
        seen_docs = set()
        sources = []
        for r in search_results:
            doc_key = (r["collection_name"], r["document_name"])
            if doc_key not in seen_docs:
                seen_docs.add(doc_key)
                sources.append({
                    "collection": r["collection_name"],
                    "document": r["document_name"],
                    "chunk_id": r["chunk_id"],
                    "text": r["text"][:300],
                    "score": round(r["score"], 3),
                })

        return {"answer": answer, "sources": sources}
