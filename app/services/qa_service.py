# -*- coding: utf-8 -*-
"""Q&A service for answering questions based on documents."""

from typing import List

from langchain_core.output_parsers import StrOutputParser

from app.config import OPENAI_QA_MODEL
from app.clients.llm_client import get_llm
from app.models.prompts import QA_PROMPT


class QAService:
    """Service for question answering."""

    def __init__(self, qdrant_manager):
        """Initialize QA service."""
        self.qdrant = qdrant_manager

    def answer_question(self, question: str, collection_names: List[str]) -> str:
        """Answer question using multi-collection search.

        Args:
            question: User question
            collection_names: List of collections to search (can be legacy sessions or new collections)

        Returns:
            Answer text
        """
        print(f"[DEBUG] Question: {question}")
        print(f"[DEBUG] Searching collections: {collection_names}")

        # Search across multiple collections
        search_results = self.qdrant.search_multi_collection(question, collection_names, k=5)

        print(f"[DEBUG] Found {len(search_results)} relevant chunks")

        if not search_results:
            return "В текущих документах нет данных для ответа."

        # Generate answer using LLM
        qa_llm = get_llm(OPENAI_QA_MODEL, temperature=0.2)
        chain = QA_PROMPT | qa_llm | StrOutputParser()

        # Format context from chunks with collection citations
        context = "\n\n".join(
            [
                f"[Collection: {r['collection_name']}, Document: {r['document_name']}, Chunk {r['chunk_id']}]\n{r['text']}"
                for r in search_results
            ]
        )

        result_text = f"{context}"
        answer = chain.invoke({"question": question, "results": result_text})

        print(f"[DEBUG] Generated answer")

        return answer
