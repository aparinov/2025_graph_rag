# -*- coding: utf-8 -*-
"""Q&A service for answering questions based on documents."""

from typing import Tuple, List

from langchain_core.output_parsers import StrOutputParser

from app.config import OPENAI_INGEST_MODEL, OPENAI_QA_MODEL
from app.clients.llm_client import get_llm
from app.models.prompts import QA_PROMPT, RELATION_PROMPT
from graph_builder import build_graph_html, extract_relations_from_chunks


class QAService:
    """Service for question answering."""

    def __init__(self, qdrant_manager):
        """Initialize QA service."""
        self.qdrant = qdrant_manager

    def answer_question(
        self, question: str, collection_names: List[str]
    ) -> Tuple[str, str]:
        """Answer question using multi-collection search and return answer + graph HTML.

        Args:
            question: User question
            collection_names: List of collections to search (can be legacy sessions or new collections)

        Returns:
            Tuple of (answer text, graph HTML)
        """
        print(f"[DEBUG] Question: {question}")
        print(f"[DEBUG] Searching collections: {collection_names}")

        # 1. Search across multiple collections
        search_results = self.qdrant.search_multi_collection(question, collection_names, k=5)

        print(f"[DEBUG] Found {len(search_results)} relevant chunks")

        if not search_results:
            return "В текущих документах нет данных для ответа.", ""

        # 2. Generate answer FIRST using LLM (fast response to user)
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

        # 3. Now process graph in background (slower operation)
        # Collect all entities from retrieved chunks
        all_entities = {}
        for chunk_data in search_results:
            if "entities" in chunk_data and chunk_data["entities"]:
                for ent in chunk_data["entities"]:
                    ent_key = f"{ent['name']}::{ent['type']}"
                    if ent_key not in all_entities:
                        all_entities[ent_key] = ent

        entities = list(all_entities.values())

        # 4. Extract relations from chunks (LLM call - can be slow)
        llm = get_llm(OPENAI_INGEST_MODEL, temperature=0.0)
        relations = extract_relations_from_chunks(search_results, llm, RELATION_PROMPT)

        print(f"[DEBUG] Extracted {len(entities)} entities, {len(relations)} relations")

        # 5. Build graph HTML
        graph_html = build_graph_html(entities, relations)

        return answer, graph_html

    def answer_question_fast(self, question: str, collection_names: List[str]) -> str:
        """Quick answer generation without graph processing.

        Args:
            question: User question
            collection_names: List of collections to search

        Returns:
            Answer text
        """
        print(f"[DEBUG] Question: {question}")

        search_results = self.qdrant.search_multi_collection(question, collection_names, k=5)
        print(f"[DEBUG] Found {len(search_results)} relevant chunks")

        if not search_results:
            return "В текущих документах нет данных для ответа."

        qa_llm = get_llm(OPENAI_QA_MODEL, temperature=0.2)
        chain = QA_PROMPT | qa_llm | StrOutputParser()

        context = "\n\n".join(
            [
                f"[Collection: {r['collection_name']}, Document: {r['document_name']}, Chunk {r['chunk_id']}]\n{r['text']}"
                for r in search_results
            ]
        )

        answer = chain.invoke({"question": question, "results": context})
        print(f"[DEBUG] Generated answer")

        return answer

    def generate_graph_from_question(self, question: str, collection_names: List[str]) -> str:
        """Generate graph HTML from question context.

        Args:
            question: User question
            collection_names: List of collections to search

        Returns:
            Graph HTML
        """
        search_results = self.qdrant.search_multi_collection(question, collection_names, k=5)

        if not search_results:
            return ""

        # Collect entities
        all_entities = {}
        for chunk_data in search_results:
            if "entities" in chunk_data and chunk_data["entities"]:
                for ent in chunk_data["entities"]:
                    ent_key = f"{ent['name']}::{ent['type']}"
                    if ent_key not in all_entities:
                        all_entities[ent_key] = ent

        entities = list(all_entities.values())

        # Extract relations
        llm = get_llm(OPENAI_INGEST_MODEL, temperature=0.0)
        relations = extract_relations_from_chunks(search_results, llm, RELATION_PROMPT)

        print(f"[DEBUG] Graph: {len(entities)} entities, {len(relations)} relations")

        return build_graph_html(entities, relations)
