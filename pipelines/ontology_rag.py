#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import asyncio
from pathlib import Path
from typing import List
from contextlib import closing

from dotenv import load_dotenv
from loguru import logger
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# LangChain / loaders
from langchain_community.document_loaders import UnstructuredPDFLoader  # requires `unstructured`
from langchain_text_splitters import CharacterTextSplitter  # or RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

# Neo4j / RDF / GraphRAG
from neo4j import Driver, GraphDatabase
from rdflib import Graph
from rdflib.namespace import OWL, RDF, RDFS

from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.experimental.components.schema import (
    SchemaBuilder,
    SchemaEntity,
    SchemaProperty,
    SchemaRelation,
)
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import FixedSizeSplitter
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.retrievers import VectorRetriever
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.indexes import create_vector_index, upsert_vector

import nest_asyncio
nest_asyncio.apply()

# ----------------------------
# Configuration
# ----------------------------
load_dotenv(override=True)
ROOT_PATH = Path().resolve()
ENV_FILE = ROOT_PATH / ".env"

class Settings(BaseSettings):
    database_url: str = Field(description="Database connection string", alias="NEO4J_URI")
    api_key: str = Field(description="API key for external service", alias="OPENAI_API_KEY")
    user: str = Field(description="DB Username credentials", alias="NEO4J_USER")
    password: str = Field(description="DB password credentials", alias="NEO4J_PASSWORD")

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra='ignore'
    )

settings = Settings()

# Constants
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
ONTOLOGY_FILE = ROOT_PATH / "pharma_ontology.ttl"
VECTOR_STORE_NAME = "pharma_embeddings_store"
EMBEDDING_DIMENSION = 1536  # text-embedding-3-small

# ----------------------------
# PDF Loading
# ----------------------------
def load_and_chunk_pdf(pdf_path: str) -> List[Document]:
    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    logger.info(f"Loading and chunking PDF: {pdf_path}")
    loader = UnstructuredPDFLoader(pdf_path)
    docs = loader.load()

    splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)

    for chunk in chunks:
        chunk.page_content = re.sub(r"\s+", " ", chunk.page_content).strip()

    logger.info(f"Created {len(chunks)} chunks.")
    return chunks

# ----------------------------
# Ontology Generation
# ----------------------------
def generate_ontology(text_sample: str, overwrite: bool = False) -> str:
    if ONTOLOGY_FILE.exists() and not overwrite:
        logger.info("Ontology file exists; skipping generation.")
        return ONTOLOGY_FILE.read_text(encoding="utf-8")

    logger.info("Generating RDF ontology with LLM...")
    llm = ChatOpenAI(
        api_key=settings.api_key,
        model="gpt-4o",
        max_tokens= 16000,
        temperature = 0.3,
    )
    prompt = f"""
    You are generating an RDF/OWL ontology in Turtle syntax for information extracted from a pharmaceutical instruction (patient leaflet / SmPC-like text).

GOALS
- Model the domain so an LLM-guided KG builder can extract consistent entities and relations from the leaflet text.
- Capture drug composition, dosage, indications, contraindications (incl. pediatric), interactions, adverse reactions, warnings/precautions, and storage.

STRICT OUTPUT
- Output ONLY valid Turtle (.ttl). No prose, no Markdown fences, no explanations.
- Do not create any individuals/instances; define schema (classes + properties) only (TBox).

PREFIXES
@prefix rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .
@prefix owl:  <http://www.w3.org/2002/07/owl#> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .
@prefix med:  <http://example.org/pharma#> .

ONTOLOGY HEADER
- Declare med: as an owl:Ontology with rdfs:label and rdfs:comment in both English (@en) and Russian (@ru) when feasible.
- Follow naming: Classes = PascalCase, properties = camelCase.

CORE CLASSES (define as owl:Class with rdfs:label @en/@ru and rdfs:comment)
- med:DrugProduct, med:ActiveSubstance, med:Excipient, med:DosageForm, med:RouteOfAdministration
- med:Strength, med:DoseInstruction, med:Frequency, med:Duration
- med:Indication, med:Contraindication, med:Warning, med:Precaution, med:Interaction, med:AdverseReaction
- med:PatientGroup, med:AgeRange, med:WeightRange, med:Condition, med:Allergy
- med:PediatricUse, med:PregnancyUse, med:LactationUse, med:HepaticImpairmentUse, med:RenalImpairmentUse, med:ElderlyUse
- med:Pharmacodynamics, med:Pharmacokinetics
- med:StorageCondition, med:ShelfLife, med:Manufacturer, med:MarketingAuthorization
- (Add narrowly-scoped subclasses if clearly present in the text, e.g., med:Ibuprofen as subclass of med:ActiveSubstance, or med:Syrup / med:Suspension as subclasses of med:DosageForm.)

DATATYPE PROPERTIES (owl:DatatypeProperty; give rdfs:domain, rdfs:range, rdfs:label @en/@ru, rdfs:comment)
- med:hasName (domain: rdfs:Resource, range: xsd:string)
- med:hasBrandName (med:DrugProduct → xsd:string)
- med:hasINN (med:ActiveSubstance → xsd:string)
- med:hasATCCode (med:DrugProduct → xsd:string)  # include only if present in the text; do NOT invent codes
- med:hasStrengthValue (med:Strength → xsd:decimal)
- med:hasStrengthUnit (med:Strength → xsd:string)  # UCUM string like "mg/mL"
- med:hasDoseAmount (med:DoseInstruction → xsd:decimal)
- med:hasDoseUnit (med:DoseInstruction → xsd:string)  # e.g., "mg", "mL", "mg/kg"
- med:hasFrequencyPerDay (med:Frequency → xsd:decimal)
- med:hasDurationDays (med:Duration → xsd:decimal)
- med:hasMaxDailyDose (med:DoseInstruction → xsd:decimal)
- med:hasMaxDailyDoseUnit (med:DoseInstruction → xsd:string)
- med:hasAgeMinMonths (med:AgeRange → xsd:integer)
- med:hasAgeMaxYears (med:AgeRange → xsd:integer)
- med:hasWeightMinKg (med:WeightRange → xsd:decimal)
- med:hasWeightMaxKg (med:WeightRange → xsd:decimal)
- med:hasStatementText (rdfs:Resource → xsd:string)  # for verbatim leaflet snippets when needed
- med:isPrescriptionOnly (med:DrugProduct → xsd:boolean)
- med:hasStorageTemperatureC (med:StorageCondition → xsd:decimal)

OBJECT PROPERTIES (owl:ObjectProperty; give rdfs:domain, rdfs:range, rdfs:label @en/@ru, rdfs:comment)
- med:containsActiveSubstance (med:DrugProduct → med:ActiveSubstance)
- med:hasExcipient (med:DrugProduct → med:Excipient)
- med:hasDosageForm (med:DrugProduct → med:DosageForm)
- med:hasRoute (med:DrugProduct → med:RouteOfAdministration)
- med:hasStrength (med:DrugProduct → med:Strength)
- med:hasDoseInstruction (med:DrugProduct → med:DoseInstruction)
- med:hasFrequency (med:DoseInstruction → med:Frequency)
- med:hasDuration (med:DoseInstruction → med:Duration)
- med:forPatientGroup (rdfs:Resource → med:PatientGroup)
- med:hasAgeRange (med:PatientGroup → med:AgeRange)
- med:hasWeightRange (med:PatientGroup → med:WeightRange)
- med:indicatedFor (med:DrugProduct → med:Indication)
- med:contraindicatedIn (med:DrugProduct → med:Contraindication)
- med:warnsAbout (med:DrugProduct → med:Warning)
- med:hasPrecaution (med:DrugProduct → med:Precaution)
- med:interactsWithDrug (med:DrugProduct → med:DrugProduct)
- med:interactsWithSubstance (med:DrugProduct → med:ActiveSubstance)
- med:mayCauseAdverseReaction (med:DrugProduct → med:AdverseReaction)
- med:hasPharmacodynamics (med:DrugProduct → med:Pharmacodynamics)
- med:hasPharmacokinetics (med:DrugProduct → med:Pharmacokinetics)
- med:hasStorageCondition (med:DrugProduct → med:StorageCondition)
- med:hasShelfLife (med:DrugProduct → med:ShelfLife)
- med:hasManufacturer (med:DrugProduct → med:Manufacturer)
- med:hasMarketingAuthorization (med:DrugProduct → med:MarketingAuthorization)
- med:hasUseGuidance (med:DrugProduct → med:PediatricUse / med:PregnancyUse / med:LactationUse / med:HepaticImpairmentUse / med:RenalImpairmentUse / med:ElderlyUse)

MODELING RULES
- Use only classes/properties supported by the text sample; omit anything not evidenced in the text (no hallucinated codes, no random subclasses).
- For pediatric guidance, represent age/weight limitations via med:PatientGroup with linked med:AgeRange / med:WeightRange and bind relevant DoseInstruction/Contraindication/Warning to that group.
- Always provide rdfs:domain and rdfs:range for every property.
- Provide rdfs:label in both English (@en) and Russian (@ru) when the text is Russian; otherwise at least @en.
- Use skos:altLabel for clear synonyms (e.g., “ibuprofen” vs. “ibuprofenum”).
- Prefer xsd:decimal for numeric quantities; keep units as strings (UCUM-style) in dedicated unit properties.
- Do not emit individuals/instance data; do not include example triples; no blank nodes.

INPUT TEXT
Between <TEXT SAMPLE> tags is the input leaflet text. Infer only what the text justifies.

<OUTPUT>
Return ONLY the Turtle ontology, nothing else.
</OUTPUT>

<TEXT SAMPLE>
{text_sample}
</TEXT SAMPLE>

    """

    response = llm.invoke(prompt)
    ontology_ttl = response.content
    ONTOLOGY_FILE.write_text(ontology_ttl, encoding="utf-8")
    logger.info(f"Ontology saved to {ONTOLOGY_FILE}")
    return ontology_ttl

# ----------------------------
# Neo4j Client
# ----------------------------
class Neo4jKGClient:
    def __init__(self, uri: str = "", user: str = "", password: str = ""):
        neo4j_uri = uri or settings.database_url
        neo4j_user = user or settings.user
        neo4j_password = password or settings.password
        if not all([neo4j_uri, neo4j_user, neo4j_password]):
            raise AttributeError("Missing Neo4j connection details.")
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        logger.info(f"Connected to Neo4j at: {neo4j_uri}")

    def close(self):
        self.driver.close()
        logger.info("Neo4j connection closed.")

    def __call__(self) -> Driver:
        return self.driver

# ----------------------------
# Ontology → GraphRAG schema (OLD API)
# ----------------------------
def get_local_part(uri: str) -> str:
    pos = max(uri.rfind("#"), uri.rfind("/"), uri.rfind(":"))
    return uri[pos + 1 :]

def get_properties_for_class(g: Graph, cat) -> list[SchemaProperty]:
    props: list[SchemaProperty] = []
    for dtp in g.subjects(RDFS.domain, cat):
        if (dtp, RDF.type, OWL.DatatypeProperty) in g:
            prop_name = get_local_part(dtp)
            prop_desc = str(next(g.objects(dtp, RDFS.comment), ""))
            props.append(SchemaProperty(name=prop_name, type="STRING", description=prop_desc))
    return props

def build_schema_from_ontology(g: Graph):
    builder = SchemaBuilder()
    entities: list[SchemaEntity] = []
    relations: list[SchemaRelation] = []
    triples: list[tuple[str, str, str]] = []
    classes: dict = {}
    print(g.subjects())
    # Classes → entities
    for cls in g.subjects(RDF.type, OWL.Class):
        classes[cls] = None
        label = get_local_part(cls)
        props = get_properties_for_class(g, cls)
        entities.append(
            SchemaEntity(
                label=label,
                description=str(next(g.objects(cls, RDFS.comment), "")),
                properties=props,
            )
        )

    # Object properties → relations + triples
    for op in g.subjects(RDF.type, OWL.ObjectProperty):
        rel_label = get_local_part(op)
        relations.append(
            SchemaRelation(
                label=rel_label,
                properties=[],
                description=str(next(g.objects(op, RDFS.comment), "")),
            )
        )
        doms = [get_local_part(d) for d in g.objects(op, RDFS.domain) if d in classes]
        rans = [get_local_part(r) for r in g.objects(op, RDFS.range) if r in classes]
        for d in doms:
            for r in rans:
                triples.append((d, rel_label, r))

    schema = builder.create_schema_model(
        entities=entities,
        relations=relations,
        potential_schema=triples,
    )

    logger.info(
        f"Ontology parsed → {len(schema.entities or {})} entities, "
        f"{len(schema.relations or {})} relations, "
        f"{len(schema.potential_schema)} triples."
    )
    return schema

# ----------------------------
# RAG Utilities
# ----------------------------
def get_chunk_embeddings(driver: Driver) -> list[tuple[int, list[float]]]:
    with driver.session() as session:
        result = session.run(
            """
            MATCH (c:Chunk)
            WHERE c.embedding IS NOT NULL
            RETURN id(c) AS node_id, c.embedding AS embedding
            """
        )
        return [(record["node_id"], record["embedding"]) for record in result]

# ----------------------------
# Main Pipeline
# ----------------------------
async def main():
    if not ENV_FILE.exists():
        raise RuntimeError("ERROR: .env file not found. Please create one with your OpenAI and Neo4j credentials.")

    pdf_path = "nurofen.pdf"

    # 1) Load & prep text
    chunks = load_and_chunk_pdf(pdf_path)
    full_text = " ".join(c.page_content for c in chunks)

    # 2) Ontology
    if not ONTOLOGY_FILE.exists():
        generate_ontology(full_text)
    g = Graph()
    g.parse(ONTOLOGY_FILE, format="turtle")
    print(g.print())
    neo4j_schema = build_schema_from_ontology(g)
    if not (neo4j_schema.relations and neo4j_schema.entities):
        raise ValueError("Provided schema has no relations or entities.")

    # 3) Build KG
    logger.info("Initializing KG builder pipeline...")
    llm = OpenAILLM(
        api_key=settings.api_key,
        model_name="gpt-4o",
        model_params={
            "max_tokens": 10_000,
            "response_format": {"type": "json_object"},
            "temperature": 0,
        },
    )
    splitter = FixedSizeSplitter(chunk_size=1500, chunk_overlap=150)
    embedder = OpenAIEmbeddings(model="text-embedding-3-small", api_key=settings.api_key)

    with closing(Neo4jKGClient()()) as driver:
        kg_builder = SimpleKGPipeline(
            llm=llm,
            driver=driver,                 # old pipeline uses `driver=`
            text_splitter=splitter,
            embedder=embedder,
            entities=list(neo4j_schema.entities.values()),
            relations=list(neo4j_schema.relations.values()),
            potential_schema=neo4j_schema.potential_schema,
            from_pdf=False
        )

        logger.info("Running KG builder...")
        await kg_builder.run_async(text=full_text)
        logger.info("KG building complete.")

        # 4) Vector store
    #     logger.info("Setting up vector index...")
    #     create_vector_index(
    #         driver,
    #         name=VECTOR_STORE_NAME,
    #         label="Chunk",
    #         embedding_property="embedding",
    #         dimensions=EMBEDDING_DIMENSION,
    #         similarity_fn="cosine",
    #     )

    #     embeddings_data = get_chunk_embeddings(driver)
    #     if not embeddings_data:
    #         logger.warning("No Chunk nodes with embeddings found.")
    #     else:
    #         logger.info(f"Upserting {len(embeddings_data)} vectors into '{VECTOR_STORE_NAME}'...")
    #         ok = 0
    #         for node_id, embedding in embeddings_data:
    #             upsert_vector(
    #                 driver,
    #                 node_id=node_id,
    #                 vector_index_name=VECTOR_STORE_NAME,
    #                 vector=embedding,
    #             )
    #             ok += 1
    #         logger.info(f"Upserted {ok} vectors.")

    # # 5) RAG query
    # logger.info("Initializing RAG pipeline...")
    # rag_llm = OpenAILLM(api_key=settings.api_key, model_name="gpt-4o", model_params={"temperature": 0})
    # rag_embedder = OpenAIEmbeddings(model="text-embedding-3-small", api_key=settings.api_key)

    # with closing(Neo4jKGClient()()) as rag_driver:
    #     retriever = VectorRetriever(rag_driver, VECTOR_STORE_NAME, rag_embedder)
    #     rag = GraphRAG(retriever=retriever, llm=rag_llm)

    #     query_text = "Какие противопоказания у Нурофена для детей?"
    #     logger.info(f"Executing RAG query: '{query_text}'")
    #     resp = rag.search(query_text=query_text, retriever_config={"top_k": 3})
    #     answer = getattr(resp, "answer", None) or getattr(resp, "response", None) or str(resp)

    #     print("\n" + "=" * 50)
    #     print(f"Query: {query_text}")
    #     print(f"Answer: {answer}")
    #     print("=" * 50 + "\n")

if __name__ == "__main__":
    asyncio.run(main())
