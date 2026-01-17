# -*- coding: utf-8 -*-

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from langchain_openai import OpenAIEmbeddings
from typing import List, Dict, Any, Optional
import uuid


class QdrantManager:
    def __init__(self, url: str = "http://localhost:6333"):
        self.client = QdrantClient(url=url)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.collection_name = "medical_documents"
        self._ensure_collection()

    def _ensure_collection(self):
        """Create collection if not exists"""
        collections = self.client.get_collections().collections
        if not any(c.name == self.collection_name for c in collections):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
            )

    def add_chunks(
        self,
        chunks: List[str],
        session_name: str,
        document_name: str,
        entities_by_chunk: List[List[Dict]],
    ):
        """Store chunks with embeddings and metadata"""
        points = []

        for i, (chunk, entities) in enumerate(zip(chunks, entities_by_chunk)):
            vector = self.embeddings.embed_query(chunk)

            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "text": chunk,
                    "session_name": session_name,
                    "document_name": document_name,
                    "chunk_id": i,
                    "entities": [
                        {"name": e["name"], "type": e["type"]} for e in entities
                    ],
                    "entity_count": len(entities),
                },
            )
            points.append(point)

        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query: str, session_name: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar chunks within session"""
        query_vector = self.embeddings.embed_query(query)

        search_filter = Filter(
            must=[
                FieldCondition(
                    key="session_name", match=MatchValue(value=session_name)
                )
            ]
        )

        if hasattr(self.client, "search"):
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=search_filter,
                limit=k,
                with_payload=True,
                with_vectors=False,
            )
        elif hasattr(self.client, "query_points"):
            try:
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    limit=k,
                    with_payload=True,
                    with_vectors=False,
                    query_filter=search_filter,
                )
            except TypeError:
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    limit=k,
                    with_payload=True,
                    with_vectors=False,
                    filter=search_filter,
                )
            results = response.points if hasattr(response, "points") else response
        elif hasattr(self.client, "search_points"):
            try:
                response = self.client.search_points(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    limit=k,
                    with_payload=True,
                    with_vectors=False,
                    query_filter=search_filter,
                )
            except TypeError:
                response = self.client.search_points(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    limit=k,
                    with_payload=True,
                    with_vectors=False,
                    filter=search_filter,
                )
            results = response.points if hasattr(response, "points") else response
        else:
            raise AttributeError(
                "QdrantClient has no search/query_points/search_points method"
            )

        return [
            {
                "text": hit.payload["text"],
                "document_name": hit.payload["document_name"],
                "chunk_id": hit.payload["chunk_id"],
                "entities": hit.payload.get("entities", []),
                "score": hit.score,
            }
            for hit in results
        ]

    def get_sessions(self) -> List[str]:
        """Get all unique session names"""
        sessions = set()
        offset = None

        while True:
            results, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=100,
                offset=offset,
                with_payload=["session_name"],
                with_vectors=False,
            )

            for point in results:
                sessions.add(point.payload["session_name"])

            if next_offset is None:
                break
            offset = next_offset

        return sorted(list(sessions))

    def get_documents(self, session_name: str) -> List[Dict[str, Any]]:
        """Get documents in session with stats"""
        offset = None
        doc_stats = {}

        while True:
            results, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=100,
                offset=offset,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="session_name", match=MatchValue(value=session_name)
                        )
                    ]
                ),
                with_payload=True,
                with_vectors=False,
            )

            for point in results:
                doc_name = point.payload["document_name"]
                if doc_name not in doc_stats:
                    doc_stats[doc_name] = {"name": doc_name, "chunks": 0, "entities": 0}
                doc_stats[doc_name]["chunks"] += 1
                doc_stats[doc_name]["entities"] += point.payload.get("entity_count", 0)

            if next_offset is None:
                break
            offset = next_offset

        return list(doc_stats.values())

    def delete_session(self, session_name: str):
        """Delete all points in a session"""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="session_name", match=MatchValue(value=session_name)
                    )
                ]
            ),
        )
