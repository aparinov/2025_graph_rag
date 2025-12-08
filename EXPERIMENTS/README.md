# EXPERIMENTS - Ontology Learning Experiments

This folder contains CLI-ready Python scripts converted from Google Colab notebooks for ontology learning experiments.

## Setup

Install dependencies:

```bash
pip install -r exp.requirements.txt
```

Set your OpenAI API key (required for most experiments):

```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Scripts

### 1. `llms4ol_experiment.py`

**LLMs4OL: Large Language Models for Ontology Learning**

Implements experiments from the paper "LLMs4OL" to test if LLMs can perform Ontology Learning tasks in Zero-Shot mode.

Tasks:
- Term Typing – determining the type of a concept
- Taxonomy Discovery – checking superclass/subclass relationships
- Relation Extraction – verifying head-relation-tail triples

```bash
# Run all tasks
python llms4ol_experiment.py --task all

# Run specific task
python llms4ol_experiment.py --task term_typing

# Use with medical data
python llms4ol_experiment.py --data-file data/acute_myeloid_leukemia_2025.md

# Fine-tune model
python llms4ol_experiment.py --task finetune --output-dir ./my-model

# Use OpenAI instead of local model
python llms4ol_experiment.py --use-openai
```

### 2. `ontology_cot_experiment.py`

**Chain-of-Thought vs Structured Prompting**

Compares two prompting strategies for ontology learning:
- Chain-of-Thought (CoT) – LLM reasons step-by-step
- Structured Prompting – strict format output

```bash
# Run classification experiment
python ontology_cot_experiment.py --task classify

# Generate ontology
python ontology_cot_experiment.py --task generate-ontology --output my_ontology.ttl

# Run with auto-fix for ontology errors
python ontology_cot_experiment.py --task auto-fix
```

### 3. `ontology_kg_pipeline.py`

**Complete Ontology-based Knowledge Graph Pipeline**

Full pipeline including:
1. Ontology seed + SHACL constraints
2. LLM-based triple extraction
3. Mapping to ontology and TTL serialization
4. SHACL + OWL reasoner validation
5. Optional Neo4j loading
6. Light GraphRAG with NetworkX
7. Question answering

```bash
# Run full pipeline
python ontology_kg_pipeline.py --task all

# Extract triples from documents
python ontology_kg_pipeline.py --task extract

# Validate knowledge graph
python ontology_kg_pipeline.py --task validate

# Answer questions
python ontology_kg_pipeline.py --task query "What are the symptoms of AML?"

# Load to Neo4j
python ontology_kg_pipeline.py --task neo4j --neo4j-url bolt://localhost:7687
```

### 4. `term_typing_eval.py`

**Term Typing Evaluation Experiment**

Evaluates LLM performance on term typing tasks with various prompting strategies.

```bash
# Run evaluation with all styles
python term_typing_eval.py --task evaluate --style all

# Use specific prompting style
python term_typing_eval.py --style cot
python term_typing_eval.py --style few-shot
python term_typing_eval.py --style structured

# Run ontology generation validation
python term_typing_eval.py --task ontology
```

## Test Data

The `data/` folder contains test data for the experiments:

- `acute_myeloid_leukemia_2025.md` - Russian clinical guidelines for Acute Myeloid Leukemia (AML)

All scripts automatically use this file when available, or you can specify it explicitly with `--data-file`.

## Medical Domain

These experiments use the Acute Myeloid Leukemia (AML) medical domain as test data. The ontology includes:

**Classes:**
- Disease (D) - leukemia, lymphoma, syndromes
- Procedure (P) - chemotherapy, transplantation, biopsy
- Symptom (S) - fever, bleeding, weakness
- DiagnosticTest (T) - immunophenotyping, cytogenetics, PCR
- Drug (M) - cytarabine, mercaptopurine
- AnatomicalEntity (A) - bone marrow, CNS, liver
- GeneticMarker (G) - FLT3, NPM1, RUNX1, t(8;21)
- Condition (C) - anemia, neutropenia, thrombocytopenia

**Relations:**
- hasSymptom, treatedWith, diagnosedBy
- hasGeneticMarker, affects, includesDrug
- hasDiagnosis, hasPrognosis

## Original Colab Notebooks

The original Colab notebooks are preserved with `.py` extension:
- `llms4ol_large_language_models_for_ontology_learning.py`
- `ontology_experiment.py`
- `ontology.py`
- `test.py`

These contain Colab-specific syntax (`!pip`, `!wget`, etc.) and are kept for reference.

## Requirements

Core dependencies:
- Python 3.8+
- torch, transformers, datasets
- openai
- rdflib, owlready2, pyshacl
- networkx, rapidfuzz
- pandas, scikit-learn

Optional:
- peft, bitsandbytes (for fine-tuning)
- py2neo (for Neo4j)
- pyvis (for visualization)

## Notes

1. **API Keys**: Most experiments require OpenAI API key. Set `OPENAI_API_KEY` environment variable.

2. **GPU**: Fine-tuning tasks benefit from GPU. Scripts automatically detect CUDA availability.

3. **Java**: OWLReady2's Pellet reasoner requires Java Runtime Environment (JRE).

4. **Neo4j**: For Neo4j integration, ensure Neo4j is running locally or provide connection URL.
