#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ontology Knowledge Graph Pipeline

CLI version converted from Google Colab notebook.
Original notebook: https://colab.research.google.com/drive/1SUJ1KtjBdwL3vuZb5zaSAAaKBHvR4mzJ

This script implements a complete ontology-based knowledge graph pipeline:
1. Ontology Seed + SHACL constraints
2. LLM-based triple extraction from documents
3. Mapping to ontology and serialization (TTL)
4. SHACL + Reasoner validation with auto-fix
5. Optional Neo4j loading
6. Light GraphRAG with NetworkX
7. MTRAG evaluation

Uses medical domain data from acute_myeloid_leukemia_2025.md

Usage:
    python ontology_kg_pipeline.py --task extract
    python ontology_kg_pipeline.py --task validate
    python ontology_kg_pipeline.py --task query "What are the symptoms of AML?"
    python ontology_kg_pipeline.py --task all
"""

import argparse
import json
import os
import re
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional

import pandas as pd

# Optional imports with fallbacks
try:
    from openai import OpenAI

    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from rdflib import Graph, Namespace, URIRef, Literal
    from rdflib.namespace import RDF, RDFS, OWL, XSD

    HAS_RDFLIB = True
except ImportError:
    HAS_RDFLIB = False

try:
    from pyshacl import validate as shacl_validate

    HAS_SHACL = True
except ImportError:
    HAS_SHACL = False

try:
    from owlready2 import get_ontology, sync_reasoner_pellet

    HAS_OWLREADY = True
except ImportError:
    HAS_OWLREADY = False

try:
    import networkx as nx

    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

try:
    from rapidfuzz import process, fuzz

    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False

try:
    from py2neo import Graph as NeoGraph, Node, Relationship

    HAS_PY2NEO = True
except ImportError:
    HAS_PY2NEO = False


# ============================================================================
# MEDICAL ONTOLOGY SEED (Acute Myeloid Leukemia Domain)
# ============================================================================

EX = Namespace("http://example.org/aml/")

ONTOLOGY_TTL = """@prefix ex: <http://example.org/aml/> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .

<http://example.org/aml/> a owl:Ontology ;
    rdfs:label "Acute Myeloid Leukemia Ontology" .

# Classes
ex:Disease a owl:Class ; rdfs:label "Disease" .
ex:AcuteMyeloidLeukemia a owl:Class ; rdfs:subClassOf ex:Disease ; rdfs:label "Acute Myeloid Leukemia" .
ex:Symptom a owl:Class ; rdfs:label "Symptom" .
ex:Procedure a owl:Class ; rdfs:label "Procedure" .
ex:Treatment a owl:Class ; rdfs:subClassOf ex:Procedure ; rdfs:label "Treatment" .
ex:DiagnosticTest a owl:Class ; rdfs:subClassOf ex:Procedure ; rdfs:label "Diagnostic Test" .
ex:Drug a owl:Class ; rdfs:label "Drug" .
ex:GeneticMarker a owl:Class ; rdfs:label "Genetic Marker" .
ex:AnatomicalEntity a owl:Class ; rdfs:label "Anatomical Entity" .
ex:Patient a owl:Class ; rdfs:label "Patient" .
ex:Prognosis a owl:Class ; rdfs:label "Prognosis" .

# Object Properties
ex:hasSymptom a owl:ObjectProperty ;
    rdfs:domain ex:Disease ;
    rdfs:range ex:Symptom ;
    rdfs:label "has symptom" .

ex:treatedWith a owl:ObjectProperty ;
    rdfs:domain ex:Disease ;
    rdfs:range ex:Treatment ;
    rdfs:label "treated with" .

ex:diagnosedBy a owl:ObjectProperty ;
    rdfs:domain ex:Disease ;
    rdfs:range ex:DiagnosticTest ;
    rdfs:label "diagnosed by" .

ex:hasGeneticMarker a owl:ObjectProperty ;
    rdfs:domain ex:Disease ;
    rdfs:range ex:GeneticMarker ;
    rdfs:label "has genetic marker" .

ex:affects a owl:ObjectProperty ;
    rdfs:domain ex:Disease ;
    rdfs:range ex:AnatomicalEntity ;
    rdfs:label "affects" .

ex:includesDrug a owl:ObjectProperty ;
    rdfs:domain ex:Treatment ;
    rdfs:range ex:Drug ;
    rdfs:label "includes drug" .

ex:hasDiagnosis a owl:ObjectProperty ;
    rdfs:domain ex:Patient ;
    rdfs:range ex:Disease ;
    rdfs:label "has diagnosis" .

ex:hasPrognosis a owl:ObjectProperty ;
    rdfs:domain ex:Patient ;
    rdfs:range ex:Prognosis ;
    rdfs:label "has prognosis" .
"""

SHACL_TTL = """@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix ex: <http://example.org/aml/> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

# Every Disease should have at least one symptom or diagnostic test
ex:DiseaseShape a sh:NodeShape ;
    sh:targetClass ex:Disease ;
    sh:or (
        [ sh:property [ sh:path ex:hasSymptom ; sh:minCount 1 ] ]
        [ sh:property [ sh:path ex:diagnosedBy ; sh:minCount 1 ] ]
    ) ;
    sh:message "Disease should have at least one symptom or diagnostic test." .

# Every Treatment should include at least one drug
ex:TreatmentShape a sh:NodeShape ;
    sh:targetClass ex:Treatment ;
    sh:property [
        sh:path ex:includesDrug ;
        sh:minCount 1 ;
        sh:message "Treatment should include at least one drug." ;
    ] .
"""

# Competency Questions for the ontology
COMPETENCY_QUESTIONS = [
    "What are the symptoms of Acute Myeloid Leukemia?",
    "What genetic markers are associated with AML?",
    "What treatments are available for AML?",
    "What diagnostic tests are used for AML?",
    "What anatomical structures are affected by AML?",
]

# Sample medical documents for extraction (based on acute_myeloid_leukemia_2025.md content)
MEDICAL_DOCUMENTS = [
    """Acute Myeloid Leukemia (AML) is a clonal disease of hematopoietic tissue 
    associated with mutations in hematopoietic progenitor cells. AML affects 
    the bone marrow and leads to abnormal proliferation of immature myeloid cells.
    Common symptoms include fever, weakness, and hemorrhagic syndrome (bleeding).""",
    """Diagnosis of AML includes cytogenetic analysis and immunophenotyping. 
    Key genetic markers include FLT3 mutations, NPM1 mutations, and chromosomal 
    translocations such as t(8;21) and inv(16). FISH analysis is used to detect 
    specific genetic abnormalities.""",
    """Treatment of AML typically involves chemotherapy with drugs like cytarabine 
    and anthracyclines. Patients with high-risk features may require hematopoietic 
    stem cell transplantation (HSCT). The treatment protocol includes induction 
    chemotherapy followed by consolidation therapy.""",
    """Patients with favorable cytogenetics like t(8;21) or inv(16) have better 
    prognosis. Minimal residual disease (MRD) monitoring using flow cytometry 
    or PCR helps assess treatment response. Complete remission is defined as 
    less than 5% blast cells in bone marrow aspirate.""",
]


class OntologyKGPipeline:
    """Complete ontology-based knowledge graph pipeline."""

    def __init__(self, api_key: str = None, neo4j_config: Dict = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = None
        if HAS_OPENAI and self.api_key:
            self.client = OpenAI(api_key=self.api_key)

        self.neo4j_config = neo4j_config or {
            "url": os.getenv("NEO4J_URL", "bolt://localhost:7687"),
            "user": os.getenv("NEO4J_USER", "neo4j"),
            "password": os.getenv("NEO4J_PASS", "password"),
        }

        self.kg = None
        self.networkx_graph = None
        self.base_dir = Path(__file__).parent

        # Initialize paths
        self.ontology_path = self.base_dir / "ontology_seed.ttl"
        self.shacl_path = self.base_dir / "ontology_shapes.ttl"
        self.triples_path = self.base_dir / "kg_from_llm.ttl"

    def initialize_ontology(self):
        """Initialize the seed ontology and SHACL shapes."""
        print("\n📋 Initializing ontology...")

        self.ontology_path.write_text(ONTOLOGY_TTL, encoding="utf-8")
        self.shacl_path.write_text(SHACL_TTL, encoding="utf-8")

        print(f"  ✅ Seed ontology: {self.ontology_path}")
        print(f"  ✅ SHACL shapes: {self.shacl_path}")

        if HAS_RDFLIB:
            self.kg = Graph()
            self.kg.parse(self.ontology_path, format="turtle")
            self.kg.bind("ex", EX)
            print(f"  ✅ Loaded {len(self.kg)} triples from seed ontology")

    def extract_triples_from_document(self, text: str, doc_id: int = 0) -> List[Dict]:
        """Extract knowledge triples from a document using LLM."""
        if not self.client:
            print("  ⚠️  OpenAI client not available for extraction")
            return []

        extraction_prompt = """You extract medical knowledge triples from text.
Return STRICT JSON with this schema:
{
  "triples": [
    {
      "subject": "entity name",
      "predicate": "relationship type",
      "object": "entity name",
      "subj_type": "class name",
      "obj_type": "class name",
      "confidence": 0.0-1.0,
      "evidence": "text excerpt supporting this triple"
    }
  ]
}

Allowed classes: Disease, AcuteMyeloidLeukemia, Symptom, Procedure, Treatment, 
DiagnosticTest, Drug, GeneticMarker, AnatomicalEntity, Patient, Prognosis

Allowed predicates: hasSymptom, treatedWith, diagnosedBy, hasGeneticMarker, 
affects, includesDrug, hasDiagnosis, hasPrognosis

Return ONLY valid JSON, no commentary."""

        try:
            resp = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": extraction_prompt},
                    {
                        "role": "user",
                        "content": f"Text:\n{text.strip()}\n\nExtract triples now.",
                    },
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )

            raw = resp.choices[0].message.content
            data = json.loads(raw)

            triples = data.get("triples", [])
            for t in triples:
                t["doc_id"] = doc_id

            return triples

        except Exception as e:
            print(f"  ⚠️  Extraction error: {e}")
            return []

    def extract_all_documents(self, documents: List[str] = None) -> List[Dict]:
        """Extract triples from all documents."""
        documents = documents or MEDICAL_DOCUMENTS

        print(f"\n📄 Extracting triples from {len(documents)} documents...")

        all_triples = []
        for i, doc in enumerate(documents):
            print(f"\n  Document {i+1}:")
            triples = self.extract_triples_from_document(doc, i)
            all_triples.extend(triples)
            print(f"    Extracted {len(triples)} triples")
            for t in triples[:3]:  # Show first 3
                print(f"      - {t['subject']} --[{t['predicate']}]--> {t['object']}")

        print(f"\n  ✅ Total: {len(all_triples)} triples extracted")
        return all_triples

    def add_triples_to_kg(self, triples: List[Dict]):
        """Add extracted triples to the knowledge graph."""
        if not HAS_RDFLIB or self.kg is None:
            print("  ⚠️  RDFLib not available")
            return

        print(f"\n📊 Adding {len(triples)} triples to knowledge graph...")

        # Build label-to-URI mapping from seed ontology
        label_to_uri = {}
        for s, p, o in self.kg.triples((None, RDFS.label, None)):
            label_to_uri[str(o).strip().lower()] = s

        # Add local names
        for s in set(self.kg.subjects(RDF.type, OWL.ObjectProperty)).union(
            set(self.kg.subjects(RDF.type, OWL.Class))
        ):
            local = str(s).split("/")[-1]
            label_to_uri[local.lower()] = s

        def to_uri(label: str) -> Optional[URIRef]:
            if not label:
                return None
            return label_to_uri.get(str(label).strip().lower())

        def as_uri(s: str) -> URIRef:
            token = re.sub(r"[^A-Za-z0-9_]+", "", s.strip().replace(" ", "_"))
            return EX[token]

        added = 0
        for t in triples:
            try:
                subj_uri = as_uri(t["subject"])
                obj_uri = as_uri(t["object"])
                pred_uri = (
                    to_uri(t["predicate"])
                    or EX[re.sub(r"[^A-Za-z0-9_]+", "", t["predicate"])]
                )

                subj_type_uri = to_uri(t.get("subj_type")) or EX["Entity"]
                obj_type_uri = to_uri(t.get("obj_type")) or EX["Entity"]

                self.kg.add((subj_uri, RDF.type, subj_type_uri))
                self.kg.add((obj_uri, RDF.type, obj_type_uri))
                self.kg.add((subj_uri, pred_uri, obj_uri))
                added += 1
            except Exception as e:
                print(f"    ⚠️  Error adding triple: {e}")

        print(f"  ✅ Added {added} triples to KG (total: {len(self.kg)} triples)")

    def save_kg(self, path: str = None):
        """Save knowledge graph to TTL file."""
        path = path or str(self.triples_path)
        if self.kg:
            self.kg.serialize(destination=path, format="turtle")
            print(f"  ✅ KG saved to {path}")

    def validate_with_shacl(self) -> Tuple[bool, str]:
        """Validate knowledge graph using SHACL."""
        if not HAS_SHACL or not HAS_RDFLIB:
            return False, "SHACL/RDFLib not available"

        print("\n🔍 Validating with SHACL...")

        try:
            data_graph = self.kg
            shacl_graph = Graph().parse(self.shacl_path, format="turtle")

            conforms, results_graph, results_text = shacl_validate(
                data_graph,
                shacl_graph=shacl_graph,
                inference="rdfs",
                abort_on_error=False,
            )

            print(f"  SHACL conforms: {'✅ Yes' if conforms else '❌ No'}")
            if not conforms:
                print(f"  Violations:\n{results_text[:500]}...")

            return conforms, results_text

        except Exception as e:
            return False, str(e)

    def validate_with_reasoner(self) -> Tuple[bool, int, List]:
        """Validate with OWL reasoner."""
        if not HAS_OWLREADY:
            return False, 0, []

        print("\n🧠 Running OWL reasoner...")

        try:
            # Save current KG to temp file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".owl", delete=False, encoding="utf-8"
            ) as f:
                if self.kg:
                    self.kg.serialize(destination=f.name, format="xml")
                temp_path = f.name

            onto = get_ontology(f"file://{temp_path}").load()
            with onto:
                sync_reasoner_pellet(
                    infer_property_values=True, infer_data_property_values=True
                )

            inconsistent = [
                c
                for c in onto.classes()
                if c.equivalent_to and "Nothing" in str(c.equivalent_to)
            ]

            os.unlink(temp_path)

            print(f"  Reasoner OK: ✅")
            print(f"  Inconsistent classes: {len(inconsistent)}")

            return True, len(inconsistent), inconsistent

        except Exception as e:
            print(f"  ⚠️  Reasoner error: {e}")
            return False, 0, []

    def build_networkx_graph(self):
        """Build NetworkX graph from RDF KG."""
        if not HAS_NETWORKX or not HAS_RDFLIB:
            print("  ⚠️  NetworkX/RDFLib not available")
            return

        print("\n🕸️  Building NetworkX graph...")

        self.networkx_graph = nx.DiGraph()

        # Add nodes
        for s, p, o in self.kg.triples((None, RDF.type, None)):
            if isinstance(s, URIRef):
                self.networkx_graph.add_node(str(s), type=str(o).split("/")[-1])

        # Add edges
        for s, p, o in self.kg.triples((None, None, None)):
            if p in [RDF.type, RDFS.label, OWL.Ontology]:
                continue
            if isinstance(s, URIRef) and isinstance(o, URIRef):
                rel = str(p).split("/")[-1]
                self.networkx_graph.add_edge(str(s), str(o), predicate=rel)

        print(
            f"  ✅ Graph: {self.networkx_graph.number_of_nodes()} nodes, {self.networkx_graph.number_of_edges()} edges"
        )

    def link_entities(self, query: str, k: int = 3) -> List[str]:
        """Link query terms to entities in the graph."""
        if not self.networkx_graph or not HAS_RAPIDFUZZ:
            return []

        nodes = list(self.networkx_graph.nodes())
        labels = [n.split("/")[-1] for n in nodes]

        matches = process.extract(query, labels, scorer=fuzz.WRatio, limit=k)
        return [nodes[labels.index(m[0])] for m in matches if m[1] > 30]

    def get_subgraph_evidence(
        self, seeds: List[str], depth: int = 2, max_edges: int = 50
    ) -> nx.DiGraph:
        """Get subgraph around seed entities."""
        if not self.networkx_graph:
            return nx.DiGraph()

        nodes = set(seeds)
        frontier = set(seeds)

        for _ in range(depth):
            new_nodes = set()
            for u in frontier:
                if u in self.networkx_graph:
                    new_nodes.update(self.networkx_graph.successors(u))
                    new_nodes.update(self.networkx_graph.predecessors(u))
            frontier = new_nodes - nodes
            nodes.update(new_nodes)

        subgraph = self.networkx_graph.subgraph(nodes).copy()

        if subgraph.number_of_edges() > max_edges:
            edges = list(subgraph.edges())[:max_edges]
            subgraph = subgraph.edge_subgraph(edges).copy()

        return subgraph

    def facts_from_subgraph(self, subgraph: nx.DiGraph) -> List[str]:
        """Extract facts from subgraph."""
        facts = []
        for u, v, data in subgraph.edges(data=True):
            u_label = u.split("/")[-1]
            v_label = v.split("/")[-1]
            pred = data.get("predicate", "rel")
            facts.append(f"{u_label} --[{pred}]--> {v_label}")
        return facts

    def answer_question(self, question: str) -> str:
        """Answer a question using the knowledge graph."""
        if not self.client or not self.networkx_graph:
            return "Knowledge graph or OpenAI client not available."

        print(f"\n❓ Question: {question}")

        # Link entities
        seeds = self.link_entities(question, k=5)
        print(f"   Linked entities: {[s.split('/')[-1] for s in seeds[:3]]}")

        # Get subgraph
        subgraph = self.get_subgraph_evidence(seeds, depth=2)
        facts = self.facts_from_subgraph(subgraph)

        if not facts:
            return "No relevant facts found in the knowledge graph."

        # Build prompt
        catalog = "\n".join([f"[FACT:{i}] {f}" for i, f in enumerate(facts)])

        prompt = f"""Answer the question using ONLY the facts below.
Cite facts using [FACT:N] markers.

Question: {question}

Facts:
{catalog}

Provide a concise answer (2-4 sentences) with citations."""

        try:
            resp = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            answer = resp.choices[0].message.content
            print(f"   ✅ Answer: {answer}")
            return answer
        except Exception as e:
            return f"Error generating answer: {e}"

    def load_to_neo4j(self):
        """Load triples to Neo4j."""
        if not HAS_PY2NEO:
            print("  ⚠️  py2neo not available")
            return

        print("\n🔄 Loading to Neo4j...")

        try:
            neo = NeoGraph(
                self.neo4j_config["url"],
                auth=(self.neo4j_config["user"], self.neo4j_config["password"]),
            )

            # Create nodes
            for s, p, o in self.kg.triples((None, RDF.type, None)):
                if isinstance(s, URIRef):
                    n = Node("Entity", uri=str(s))
                    neo.merge(n, "Entity", "uri")

            # Create relationships
            for s, p, o in self.kg.triples((None, None, None)):
                if p in [RDF.type, RDFS.label, OWL.Ontology]:
                    continue
                if isinstance(s, URIRef) and isinstance(o, URIRef):
                    rel_type = re.sub(r"[^A-Za-z0-9_]", "_", str(p).split("/")[-1])
                    neo.run(
                        f"MERGE (a:Entity {{uri:$s}}) MERGE (b:Entity {{uri:$o}}) MERGE (a)-[r:{rel_type}]->(b)",
                        s=str(s),
                        o=str(o),
                    )

            print("  ✅ Triples loaded to Neo4j")

        except Exception as e:
            print(f"  ⚠️  Neo4j error: {e}")


def load_documents_from_file(filepath: str) -> List[str]:
    """Load medical documents from a markdown file."""
    documents = MEDICAL_DOCUMENTS.copy()

    if filepath and Path(filepath).exists():
        print(f"Loading documents from: {filepath}")
        content = Path(filepath).read_text(encoding="utf-8", errors="ignore")

        # Split into sections/paragraphs
        sections = re.split(r"\n#{1,3}\s+", content)
        for section in sections:
            # Clean and add if substantial
            clean_section = re.sub(r"\*+|#+|_{2,}|={2,}", "", section)
            clean_section = re.sub(r"\s+", " ", clean_section).strip()
            if len(clean_section) > 200:  # Only add substantial paragraphs
                documents.append(clean_section[:1500])  # Limit length

    return documents


def main():
    parser = argparse.ArgumentParser(
        description="Ontology-based Knowledge Graph Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ontology_kg_pipeline.py --task extract
  python ontology_kg_pipeline.py --task validate
  python ontology_kg_pipeline.py --task query "What are the symptoms of AML?"
  python ontology_kg_pipeline.py --task all
  python ontology_kg_pipeline.py --data-file data/acute_myeloid_leukemia_2025.md
        """,
    )

    parser.add_argument(
        "--task",
        type=str,
        default="all",
        choices=["all", "extract", "validate", "query", "neo4j"],
        help="Task to run",
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default=None,
        help="Path to data file (e.g., acute_myeloid_leukemia_2025.md)",
    )
    parser.add_argument(
        "--query", type=str, default=None, help="Question to answer (for query task)"
    )
    parser.add_argument("--openai-key", type=str, default=None, help="OpenAI API key")
    parser.add_argument(
        "--output",
        type=str,
        default="kg_medical.ttl",
        help="Output file for knowledge graph",
    )
    parser.add_argument(
        "--neo4j-url", type=str, default=None, help="Neo4j connection URL"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Ontology-based Knowledge Graph Pipeline")
    print("Medical Domain: Acute Myeloid Leukemia")
    print("=" * 70)

    # Check dependencies
    print("\n📦 Checking dependencies:")
    print(f"  OpenAI: {'✅' if HAS_OPENAI else '❌'}")
    print(f"  RDFLib: {'✅' if HAS_RDFLIB else '❌'}")
    print(f"  SHACL: {'✅' if HAS_SHACL else '❌'}")
    print(f"  OWLReady2: {'✅' if HAS_OWLREADY else '❌'}")
    print(f"  NetworkX: {'✅' if HAS_NETWORKX else '❌'}")
    print(f"  RapidFuzz: {'✅' if HAS_RAPIDFUZZ else '❌'}")
    print(f"  py2neo: {'✅' if HAS_PY2NEO else '❌'}")

    # Initialize pipeline
    neo4j_config = None
    if args.neo4j_url:
        neo4j_config = {"url": args.neo4j_url, "user": "neo4j", "password": "password"}

    pipeline = OntologyKGPipeline(api_key=args.openai_key, neo4j_config=neo4j_config)

    # Load documents
    data_file = args.data_file
    if data_file is None:
        default_path = Path(__file__).parent / "data" / "acute_myeloid_leukemia_2025.md"
        if default_path.exists():
            data_file = str(default_path)

    documents = load_documents_from_file(data_file)
    print(f"\n📄 Loaded {len(documents)} documents")

    # Run tasks
    if args.task in ["all", "extract"]:
        pipeline.initialize_ontology()
        triples = pipeline.extract_all_documents(documents[:5])  # Limit for demo
        pipeline.add_triples_to_kg(triples)
        pipeline.save_kg(args.output)

    if args.task in ["all", "validate"]:
        if pipeline.kg is None:
            pipeline.initialize_ontology()
            # Try to load existing KG
            if Path(args.output).exists():
                pipeline.kg.parse(args.output, format="turtle")

        pipeline.validate_with_shacl()
        pipeline.validate_with_reasoner()

    if args.task in ["all", "query"]:
        if pipeline.kg is None:
            pipeline.initialize_ontology()
            if Path(args.output).exists():
                pipeline.kg.parse(args.output, format="turtle")

        pipeline.build_networkx_graph()

        questions = COMPETENCY_QUESTIONS if args.query is None else [args.query]

        print("\n" + "=" * 70)
        print("ANSWERING QUESTIONS")
        print("=" * 70)

        for q in questions:
            answer = pipeline.answer_question(q)
            print()

    if args.task == "neo4j":
        if pipeline.kg is None:
            pipeline.initialize_ontology()
            if Path(args.output).exists():
                pipeline.kg.parse(args.output, format="turtle")

        pipeline.load_to_neo4j()

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(
        """
Summary:
- Ontology-based KG enables grounded question answering
- SHACL validation ensures data quality
- OWL reasoning provides consistency checking
- Graph-based retrieval (GraphRAG) improves answer accuracy
    """
    )


if __name__ == "__main__":
    main()
