#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import uuid
import json
from typing import List, Dict, Any, Tuple
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from dotenv import load_dotenv

# LangChain
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
from langchain_neo4j import Neo4jGraph
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Embeddings + vector similarity
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz

# JSON parsing
import json_repair

load_dotenv(override=True)

# ----------------------------
# Configuration
# ----------------------------
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
BATCH_SIZE = 5
MAX_WORKERS = 3
SIMILARITY_THRESHOLD = 0.85
FUZZY_THRESHOLD = 85

# Medical entity types for pharmaceutical instructions
ENTITY_TYPES = {
    "Препарат", "ДействующееВещество", "Дозировка", "ПутьВведения", 
    "ФормаВыпуска", "Показание", "Противопоказание", "ГруппаПациентов", 
    "ПобочныйЭффект", "Взаимодействие", "Производитель", "УсловияХранения",
    "СрокГодности", "УсловияОтпуска"
}

RELATION_TYPES = {
    "СОДЕРЖИТ_ДВ": ("Препарат", "ДействующееВещество"),
    "ИМЕЕТ_ФОРМУ": ("Препарат", "ФормаВыпуска"),
    "ПРИМЕНЯЕТСЯ_ЧЕРЕЗ": ("Препарат", "ПутьВведения"),
    "ПОКАЗАН_ПРИ": ("Препарат", "Показание"),
    "ПРОТИВОПОКАЗАН_ПРИ": ("Препарат", "Противопоказание"),
    "ВЫЗЫВАЕТ_ПОБОЧНЫЙ_ЭФФЕКТ": ("Препарат", "ПобочныйЭффект"),
    "ДОЗИРОВКА_ДЛЯ": ("Дозировка", "ГруппаПациентов"),
    "ВЗАИМОДЕЙСТВУЕТ_С": ("Препарат", "Взаимодействие"),
    "ПРОИЗВЕДЁН": ("Препарат", "Производитель"),
}

# Neo4j setup
NEO4J_HOST = os.getenv("NEO4J_HOST", "localhost")
NEO4J_PORT = os.getenv("NEO4J_PORT", "7687")
NEO4J_URL = f"neo4j://{NEO4J_HOST}:{NEO4J_PORT}"
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# ----------------------------
# Improved Entity Extractor
# ----------------------------
class StableEntityExtractor:
    def __init__(self):
        self.llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.0,
            max_tokens=2000
        )
        
        # Simplified prompt for more stable parsing
        self.extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", """Ты эксперт по медицинским текстам. Извлеки сущности из фрагмента инструкции к лекарству.

ВАЖНО: Отвечай ТОЛЬКО валидным JSON массивом без дополнительного текста.

Типы сущностей: {entity_types}

Формат ответа:
[
  {{"name": "название", "type": "тип", "context": "краткий контекст"}},
  {{"name": "другое название", "type": "другой тип", "context": "контекст"}}
]

Если сущностей нет, верни: []"""),
            ("user", "Текст:\n{text}")
        ])
        
        self.chain = self.extraction_prompt | self.llm | StrOutputParser()
    
    def extract_batch(self, chunks: List[str]) -> List[List[Dict[str, Any]]]:
        """Extract entities from multiple chunks in parallel"""
        results = []
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(self._extract_single, chunk): i 
                for i, chunk in enumerate(chunks)
            }
            
            # Initialize results list with correct size
            results = [[] for _ in range(len(chunks))]
            
            for future in as_completed(futures):
                chunk_idx = futures[future]
                try:
                    entities = future.result()
                    results[chunk_idx] = entities
                    print(f"✓ Chunk {chunk_idx + 1}: {len(entities)} entities")
                except Exception as e:
                    print(f"✗ Chunk {chunk_idx + 1} failed: {e}")
                    results[chunk_idx] = []
        
        return results
    
    def _extract_single(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities from a single chunk"""
        try:
            response = self.chain.invoke({
                "text": text[:1500],  # Limit text length
                "entity_types": ", ".join(ENTITY_TYPES)
            })
            
            # Clean the response
            response = response.strip()
            if response.startswith("```"):
                response = re.sub(r"```(?:json)?\s*", "", response)
                response = re.sub(r"\s*```", "", response)
            
            entities = json_repair.loads(response)
            
            if not isinstance(entities, list):
                return []
            
            # Validate and clean entities
            valid_entities = []
            for entity in entities:
                if (isinstance(entity, dict) and 
                    "name" in entity and 
                    "type" in entity and
                    entity["type"] in ENTITY_TYPES):
                    
                    valid_entities.append({
                        "name": str(entity["name"]).strip(),
                        "type": str(entity["type"]).strip(),
                        "context": str(entity.get("context", "")).strip()[:200]
                    })
            
            return valid_entities
            
        except Exception as e:
            print(f"Entity extraction error: {e}")
            return []

# ----------------------------
# Fast Entity Deduplication
# ----------------------------
class FastDeduplicator:
    def __init__(self):
        try:
            self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        except Exception as e:
            print(f"Warning: Could not initialize embeddings: {e}")
            self.embeddings = None
    
    def deduplicate(self, all_entities: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
        """Fast deduplication using embeddings and fuzzy matching"""
        if not all_entities:
            return [], {}
        
        print(f"🔄 Deduplicating {len(all_entities)} entities...")
        
        # Group by type first for efficiency
        type_groups = defaultdict(list)
        for i, entity in enumerate(all_entities):
            entity["_idx"] = i  # Track original index
            type_groups[entity["type"]].append(entity)
        
        canonical_entities = []
        entity_mapping = {}
        
        for entity_type, entities in type_groups.items():
            if len(entities) == 1:
                # No duplicates possible
                canonical = self._create_canonical(entities)
                canonical_entities.append(canonical)
                entity_mapping[entities[0]["_idx"]] = canonical["id"]
                continue
            
            # Deduplicate within type
            type_canonical, type_mapping = self._deduplicate_by_type(entities)
            canonical_entities.extend(type_canonical)
            entity_mapping.update(type_mapping)
        
        print(f"✓ Reduced to {len(canonical_entities)} canonical entities")
        return canonical_entities, entity_mapping
    
    def _deduplicate_by_type(self, entities: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
        """Deduplicate entities of the same type"""
        if len(entities) <= 1:
            canonical = self._create_canonical(entities) if entities else None
            mapping = {entities[0]["_idx"]: canonical["id"]} if canonical else {}
            return [canonical] if canonical else [], mapping
        
        # Quick fuzzy grouping first
        groups = self._fuzzy_group(entities)
        
        # For each fuzzy group, use embeddings if needed and available
        final_groups = []
        for group in groups:
            if len(group) <= 3 or not self.embeddings:
                final_groups.append(group)
            else:
                # Use embeddings for large groups
                try:
                    semantic_groups = self._semantic_group(group)
                    final_groups.extend(semantic_groups)
                except Exception as e:
                    print(f"Semantic grouping failed: {e}, falling back to fuzzy")
                    final_groups.append(group)
        
        # Create canonical entities
        canonical_entities = []
        entity_mapping = {}
        
        for group in final_groups:
            canonical = self._create_canonical(group)
            canonical_entities.append(canonical)
            for entity in group:
                entity_mapping[entity["_idx"]] = canonical["id"]
        
        return canonical_entities, entity_mapping
    
    def _fuzzy_group(self, entities: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Group entities using fuzzy string matching"""
        groups = []
        used = set()
        
        for i, entity in enumerate(entities):
            if i in used:
                continue
            
            group = [entity]
            used.add(i)
            
            for j, other in enumerate(entities[i+1:], i+1):
                if j in used:
                    continue
                
                if fuzz.token_sort_ratio(entity["name"], other["name"]) >= FUZZY_THRESHOLD:
                    group.append(other)
                    used.add(j)
            
            groups.append(group)
        
        return groups
    
    def _semantic_group(self, entities: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Group entities using semantic embeddings"""
        if len(entities) <= 1 or not self.embeddings:
            return [entities]
        
        # Create embeddings
        texts = [f"{e['name']} {e['context']}" for e in entities]
        embeddings = self.embeddings.embed_documents(texts)
        embeddings = np.array(embeddings)
        
        # Compute similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        
        # Simple clustering based on similarity threshold
        groups = []
        used = set()
        
        for i in range(len(entities)):
            if i in used:
                continue
            
            group = [entities[i]]
            used.add(i)
            
            for j in range(i+1, len(entities)):
                if j in used:
                    continue
                
                if similarity_matrix[i][j] >= SIMILARITY_THRESHOLD:
                    group.append(entities[j])
                    used.add(j)
            
            groups.append(group)
        
        return groups
    
    def _create_canonical(self, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create canonical entity from group"""
        if not entities:
            return None
        
        # Choose best name (most common or longest)
        names = [e["name"] for e in entities]
        name_counts = Counter(names)
        best_name = max(name_counts.items(), key=lambda x: (x[1], len(x[0])))[0]
        
        # Merge contexts
        contexts = [e["context"] for e in entities if e["context"]]
        merged_context = " | ".join(set(contexts))[:500]
        
        return {
            "id": f"ent_{uuid.uuid4().hex[:12]}",
            "name": best_name,
            "type": entities[0]["type"],
            "context": merged_context,
            "member_count": len(entities)
        }

# ----------------------------
# Relation Extractor
# ----------------------------
class RelationExtractor:
    def __init__(self):
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0)
        
        self.relation_prompt = ChatPromptTemplate.from_messages([
            ("system", """Определи отношения между сущностями в тексте.

Возможные отношения: {relations}

Формат ответа - JSON массив:
[
  {{"source": "название1", "target": "название2", "relation": "ТИП_ОТНОШЕНИЯ", "confidence": 0.9}},
  {{"source": "название2", "target": "название3", "relation": "ДРУГОЙ_ТИП", "confidence": 0.8}}
]

Если отношений нет, верни: []"""),
            ("user", "Текст: {text}\n\nСущности: {entities}")
        ])
        
        self.chain = self.relation_prompt | self.llm | StrOutputParser()
    
    def extract_relations(self, chunks: List[str], entities_by_chunk: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Extract relations from chunks"""
        all_relations = []
        
        for chunk_idx, (chunk_text, chunk_entities) in enumerate(zip(chunks, entities_by_chunk)):
            if len(chunk_entities) < 2:
                continue
            
            try:
                relations = self._extract_chunk_relations(chunk_text, chunk_entities)
                for rel in relations:
                    rel["chunk_idx"] = chunk_idx
                all_relations.extend(relations)
                print(f"✓ Chunk {chunk_idx + 1}: {len(relations)} relations")
            except Exception as e:
                print(f"✗ Relation extraction failed for chunk {chunk_idx + 1}: {e}")
        
        return all_relations
    
    def _extract_chunk_relations(self, text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract relations from a single chunk"""
        entity_names = [e["name"] for e in entities]
        entity_list = ", ".join(entity_names)
        
        response = self.chain.invoke({
            "text": text[:1000],
            "entities": entity_list,
            "relations": ", ".join(RELATION_TYPES.keys())
        })
        
        # Clean and parse response
        response = response.strip()
        if response.startswith("```"):
            response = re.sub(r"```(?:json)?\s*", "", response)
            response = re.sub(r"\s*```", "", response)
        
        try:
            relations = json_repair.loads(response)
        except Exception as e:
            print(f"JSON parsing error: {e}")
            return []
        
        if not isinstance(relations, list):
            return []
        
        valid_relations = []
        for rel in relations:
            if (isinstance(rel, dict) and
                all(k in rel for k in ["source", "target", "relation"]) and
                rel["relation"] in RELATION_TYPES and
                rel.get("confidence", 0) > 0.5):
                
                valid_relations.append({
                    "source": str(rel["source"]).strip(),
                    "target": str(rel["target"]).strip(),
                    "relation": str(rel["relation"]).strip(),
                    "confidence": float(rel.get("confidence", 0.7))
                })
        
        return valid_relations

# ----------------------------
# Neo4j Writer
# ----------------------------
class Neo4jWriter:
    def __init__(self):
        self.graph = Neo4jGraph(
            url=NEO4J_URL,
            username=NEO4J_USER,
            password=NEO4J_PASSWORD,
            refresh_schema=False
        )
    
    def write_graph(self, document_name: str, entities: List[Dict[str, Any]], 
                   relations: List[Dict[str, Any]], chunk_count: int):
        """Write entities and relations to Neo4j efficiently"""
        print(f"💾 Writing to Neo4j: {len(entities)} entities, {len(relations)} relations")
        
        # Create document node
        self.graph.query(
            "MERGE (d:Document {name: $name}) SET d.chunks = $chunks, d.updated = timestamp()",
            params={"name": document_name, "chunks": chunk_count}
        )
        
        # Batch create entities
        self._batch_create_entities(entities, document_name)
        
        # Batch create relations
        self._batch_create_relations(relations)
        
        print("✓ Graph written to Neo4j")
    
    def _batch_create_entities(self, entities: List[Dict[str, Any]], document_name: str):
        """Create entities in batches"""
        for i in range(0, len(entities), BATCH_SIZE):
            batch = entities[i:i+BATCH_SIZE]
            
            # Create nodes
            for entity in batch:
                label = entity["type"].replace(" ", "_")
                try:
                    self.graph.query(
                        f"""
                        MERGE (e:`{label}` {{id: $id}})
                        SET e.name = $name,
                            e.type = $type,
                            e.context = $context,
                            e.member_count = $member_count
                        """,
                        params={
                            "id": entity["id"],
                            "name": entity["name"],
                            "type": entity["type"],
                            "context": entity["context"],
                            "member_count": entity.get("member_count", 1)
                        }
                    )
                except Exception as e:
                    print(f"Error creating entity {entity['name']}: {e}")
            
            # Link to document
            entity_ids = [e["id"] for e in batch]
            try:
                self.graph.query(
                    """
                    MATCH (e) WHERE e.id IN $ids
                    MATCH (d:Document {name: $doc_name})
                    MERGE (e)-[:APPEARS_IN]->(d)
                    """,
                    params={"ids": entity_ids, "doc_name": document_name}
                )
            except Exception as e:
                print(f"Error linking entities to document: {e}")
    
    def _batch_create_relations(self, relations: List[Dict[str, Any]]):
        """Create relations in batches"""
        for i in range(0, len(relations), BATCH_SIZE):
            batch = relations[i:i+BATCH_SIZE]
            
            for rel in batch:
                try:
                    self.graph.query(
                        f"""
                        MATCH (a {{name: $source}}), (b {{name: $target}})
                        MERGE (a)-[r:`{rel['relation']}`]->(b)
                        SET r.confidence = $confidence
                        """,
                        params={
                            "source": rel["source"],
                            "target": rel["target"],
                            "confidence": rel["confidence"]
                        }
                    )
                except Exception as e:
                    print(f"Error creating relation {rel['source']} -> {rel['target']}: {e}")

# ----------------------------
# Main Pipeline
# ----------------------------
class ImprovedPipeline:
    def __init__(self):
        self.entity_extractor = StableEntityExtractor()
        self.deduplicator = FastDeduplicator()
        self.relation_extractor = RelationExtractor()
        self.neo4j_writer = Neo4jWriter()
    
    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Main processing pipeline"""
        start_time = time.time()
        document_name = os.path.splitext(os.path.basename(pdf_path))[0]
        
        print(f"🚀 Processing: {document_name}")
        
        try:
            # 1. Load and chunk document
            chunks = self._load_and_chunk(pdf_path)
            chunk_texts = [c.page_content for c in chunks]
            
            # 2. Extract entities
            entities_by_chunk = self.entity_extractor.extract_batch(chunk_texts)
            all_entities = [e for chunk_entities in entities_by_chunk for e in chunk_entities]
            
            # 3. Deduplicate entities
            canonical_entities, entity_mapping = self.deduplicator.deduplicate(all_entities)
            
            # 4. Extract relations
            relations = self.relation_extractor.extract_relations(chunk_texts, entities_by_chunk)
            
            # 5. Write to Neo4j
            self.neo4j_writer.write_graph(document_name, canonical_entities, relations, len(chunks))
            
            # Results
            processing_time = time.time() - start_time
            result = {
                "document": document_name,
                "chunks": len(chunks),
                "entities": len(canonical_entities),
                "relations": len(relations),
                "processing_time": round(processing_time, 2)
            }
            
            print(f"✅ Completed in {processing_time:.1f}s: {len(canonical_entities)} entities, {len(relations)} relations")
            return result
            
        except Exception as e:
            print(f"❌ Processing failed: {e}")
            raise e
    
    def _load_and_chunk(self, pdf_path: str) -> List[Document]:
        """Load PDF and split into chunks"""
        loader = UnstructuredPDFLoader(pdf_path)
        docs = loader.load()
        
        splitter = CharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        
        chunks = splitter.split_documents(docs)
        
        # Clean chunk texts
        for chunk in chunks:
            chunk.page_content = re.sub(r'\s+', ' ', chunk.page_content).strip()
        
        return chunks

# ----------------------------
# CLI Interface
# ----------------------------
def main():
    pipeline = ImprovedPipeline()
    result = pipeline.process_pdf('nurofen.pdf')
    
    print("\n📊 Results:")
    for key, value in result.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main()
