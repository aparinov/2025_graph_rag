#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Term Typing Evaluation Experiment

CLI version converted from Google Colab notebook.
This script evaluates LLM performance on term typing tasks using various datasets
and prompting strategies.

Includes:
- GeoNames-style term typing
- Medical domain term typing
- LLMs4OL Challenge dataset loading
- Comparison of prompting strategies (CoT, Structured, Few-shot)

Uses medical domain data from acute_myeloid_leukemia_2025.md

Usage:
    python term_typing_eval.py --task evaluate
    python term_typing_eval.py --task medical --data-file data/acute_myeloid_leukemia_2025.md
    python term_typing_eval.py --style cot
    python term_typing_eval.py --style few-shot
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional

import pandas as pd

# Optional imports
try:
    from openai import OpenAI

    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from owlready2 import get_ontology, sync_reasoner_pellet

    HAS_OWLREADY = True
except ImportError:
    HAS_OWLREADY = False

try:
    from rdflib import Graph

    HAS_RDFLIB = True
except ImportError:
    HAS_RDFLIB = False


# ============================================================================
# MEDICAL DOMAIN TERM TYPING (from acute_myeloid_leukemia_2025.md)
# ============================================================================

# Medical ontology top-level classes based on the AML clinical guidelines
MEDICAL_TOP_CLASSES = {
    "D": [
        "disease",
        "syndrome",
        "leukemia",
        "lymphoma",
        "болезнь",
        "синдром",
        "лейкоз",
    ],
    "P": [
        "procedure",
        "treatment",
        "therapy",
        "transplantation",
        "процедура",
        "терапия",
        "трансплантация",
    ],
    "S": [
        "symptom",
        "sign",
        "fever",
        "bleeding",
        "симптом",
        "лихорадка",
        "кровотечение",
    ],
    "T": ["test", "diagnostic", "analysis", "исследование", "анализ", "диагностика"],
    "M": ["drug", "medication", "medicine", "препарат", "лекарство", "медикамент"],
    "A": [
        "anatomy",
        "organ",
        "tissue",
        "bone",
        "анатомия",
        "орган",
        "ткань",
        "костный",
    ],
    "G": ["gene", "marker", "mutation", "translocation", "ген", "маркер", "мутация"],
    "C": [
        "condition",
        "state",
        "anemia",
        "neutropenia",
        "состояние",
        "анемия",
        "нейтропения",
    ],
}

MEDICAL_VALID_LETTERS = list(MEDICAL_TOP_CLASSES.keys())
MEDICAL_CLASS_DESCRIPTIONS = "; ".join(
    [f"{k}: {', '.join(v[:3])}" for k, v in MEDICAL_TOP_CLASSES.items()]
)

# Medical term samples for evaluation
MEDICAL_TERM_SAMPLES = [
    {"term": "острый миелоидный лейкоз", "gold_type": "D", "gold_full": "Disease"},
    {"term": "acute myeloid leukemia", "gold_type": "D", "gold_full": "Disease"},
    {"term": "химиотерапия", "gold_type": "P", "gold_full": "Procedure"},
    {"term": "chemotherapy", "gold_type": "P", "gold_full": "Procedure"},
    {"term": "лихорадка", "gold_type": "S", "gold_full": "Symptom"},
    {"term": "fever", "gold_type": "S", "gold_full": "Symptom"},
    {"term": "иммунофенотипирование", "gold_type": "T", "gold_full": "DiagnosticTest"},
    {"term": "immunophenotyping", "gold_type": "T", "gold_full": "DiagnosticTest"},
    {"term": "цитарабин", "gold_type": "M", "gold_full": "Drug"},
    {"term": "cytarabine", "gold_type": "M", "gold_full": "Drug"},
    {"term": "костный мозг", "gold_type": "A", "gold_full": "AnatomicalEntity"},
    {"term": "bone marrow", "gold_type": "A", "gold_full": "AnatomicalEntity"},
    {"term": "FLT3", "gold_type": "G", "gold_full": "GeneticMarker"},
    {"term": "t(8;21)", "gold_type": "G", "gold_full": "GeneticMarker"},
    {"term": "нейтропения", "gold_type": "C", "gold_full": "Condition"},
    {"term": "neutropenia", "gold_type": "C", "gold_full": "Condition"},
    {"term": "тромбоцитопения", "gold_type": "C", "gold_full": "Condition"},
    {"term": "thrombocytopenia", "gold_type": "C", "gold_full": "Condition"},
    {
        "term": "трансплантация гемопоэтических стволовых клеток",
        "gold_type": "P",
        "gold_full": "Procedure",
    },
    {
        "term": "hematopoietic stem cell transplantation",
        "gold_type": "P",
        "gold_full": "Procedure",
    },
    {"term": "миелодиспластический синдром", "gold_type": "D", "gold_full": "Disease"},
    {"term": "myelodysplastic syndrome", "gold_type": "D", "gold_full": "Disease"},
    {"term": "геморрагический синдром", "gold_type": "S", "gold_full": "Symptom"},
    {"term": "hemorrhagic syndrome", "gold_type": "S", "gold_full": "Symptom"},
    {"term": "RUNX1-RUNX1T1", "gold_type": "G", "gold_full": "GeneticMarker"},
    {"term": "NPM1", "gold_type": "G", "gold_full": "GeneticMarker"},
    {"term": "ПЦР", "gold_type": "T", "gold_full": "DiagnosticTest"},
    {"term": "PCR", "gold_type": "T", "gold_full": "DiagnosticTest"},
    {"term": "FISH", "gold_type": "T", "gold_full": "DiagnosticTest"},
    {"term": "спинномозговая пункция", "gold_type": "P", "gold_full": "Procedure"},
]

# Few-shot examples for medical domain
MEDICAL_FEW_SHOT_EXAMPLES = """Examples:
Term: chemotherapy -> P
Term: leukemia -> D
Term: fever -> S
Term: bone marrow biopsy -> T
Term: cytarabine -> M
Term: spleen -> A
Term: FLT3 mutation -> G
Term: anemia -> C
"""


def load_additional_terms_from_file(filepath: str) -> List[Dict]:
    """Extract additional terms from the medical guidelines file."""
    additional_terms = []

    if filepath and Path(filepath).exists():
        content = Path(filepath).read_text(encoding="utf-8", errors="ignore")

        # Extract abbreviations with definitions
        abbreviations = re.findall(r"\b([А-ЯA-Z]{2,8})\s+[–—-]\s+([^\n,]+)", content)

        for abbr, definition in abbreviations:
            # Determine type based on definition
            def_lower = definition.lower()
            gold_type = "D"  # Default to disease

            if any(
                kw in def_lower
                for kw in [
                    "терапия",
                    "трансплантация",
                    "лечение",
                    "therapy",
                    "treatment",
                ]
            ):
                gold_type = "P"
            elif any(
                kw in def_lower for kw in ["исследование", "анализ", "test", "analysis"]
            ):
                gold_type = "T"
            elif any(
                kw in def_lower
                for kw in ["препарат", "лекарство", "drug", "medication"]
            ):
                gold_type = "M"
            elif any(
                kw in def_lower
                for kw in ["ген", "мутация", "gene", "mutation", "marker"]
            ):
                gold_type = "G"
            elif any(
                kw in def_lower for kw in ["симптом", "признак", "symptom", "sign"]
            ):
                gold_type = "S"
            elif any(
                kw in def_lower for kw in ["орган", "ткань", "мозг", "organ", "tissue"]
            ):
                gold_type = "A"
            elif any(kw in def_lower for kw in ["пения", "цитоз", "emia", "cytosis"]):
                gold_type = "C"

            additional_terms.append(
                {"term": abbr, "gold_type": gold_type, "gold_full": definition[:50]}
            )

    return additional_terms


class TermTypingEvaluator:
    """Evaluator for term typing experiments."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = None
        if HAS_OPENAI and self.api_key:
            self.client = OpenAI(api_key=self.api_key)

    def predict_structured(self, term: str, context: str = None) -> str:
        """Predict using structured prompting."""
        if not self.client:
            return "?"

        prompt = f"""Classify the medical term into ONE top class.
Classes:
{MEDICAL_CLASS_DESCRIPTIONS}

Choose exactly one LETTER among [{', '.join(MEDICAL_VALID_LETTERS)}].
Term: {term}
{"Context: " + context if context else ""}
OUTPUT ONLY THE LETTER."""

        try:
            resp = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            result = resp.choices[0].message.content.strip().upper()
            # Extract single letter
            match = re.search(r"[DPSTMAGC]", result)
            return match.group(0) if match else "?"
        except Exception as e:
            print(f"  Error: {e}")
            return "?"

    def predict_cot(self, term: str, context: str = None) -> str:
        """Predict using Chain-of-Thought prompting."""
        if not self.client:
            return "?"

        prompt = f"""Classify the medical term into ONE top class.
Classes:
{MEDICAL_CLASS_DESCRIPTIONS}

Choose exactly one LETTER among [{', '.join(MEDICAL_VALID_LETTERS)}].
Term: {term}
{"Context: " + context if context else ""}

Think step-by-step:
1. What does this term refer to?
2. Is it a disease, procedure, symptom, test, drug, anatomy, gene, or condition?
3. Which letter best represents it?

Think briefly, then OUTPUT ONLY the final LETTER."""

        try:
            resp = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            result = resp.choices[0].message.content.strip().upper()
            match = re.search(r"[DPSTMAGC]", result)
            return match.group(0) if match else "?"
        except Exception as e:
            print(f"  Error: {e}")
            return "?"

    def predict_few_shot(self, term: str, context: str = None) -> str:
        """Predict using few-shot prompting."""
        if not self.client:
            return "?"

        prompt = f"""Classify the medical term into ONE top class.
Classes: D(disease), P(procedure), S(symptom), T(test), M(drug), A(anatomy), G(gene/marker), C(condition).
Choose exactly one LETTER among [D,P,S,T,M,A,G,C].

{MEDICAL_FEW_SHOT_EXAMPLES}
Term: {term}
{"Context: " + context if context else ""}
OUTPUT ONLY the LETTER."""

        try:
            resp = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            result = resp.choices[0].message.content.strip().upper()
            match = re.search(r"[DPSTMAGC]", result)
            return match.group(0) if match else "?"
        except Exception as e:
            print(f"  Error: {e}")
            return "?"

    def evaluate(self, df: pd.DataFrame, style: str = "structured") -> Dict[str, Any]:
        """Evaluate on a dataframe of terms."""
        predict_fn = {
            "structured": self.predict_structured,
            "cot": self.predict_cot,
            "few-shot": self.predict_few_shot,
        }.get(style, self.predict_structured)

        print(f"\nEvaluating with {style.upper()} prompting...")
        print("-" * 60)

        preds = []
        for _, row in df.iterrows():
            pred = predict_fn(row["term"], row.get("context"))
            preds.append(pred)
            status = "✓" if pred == row["gold_type"] else "✗"
            print(
                f"  {status} {row['term'][:40]:40s} | Gold: {row['gold_type']} | Pred: {pred}"
            )

        # Calculate metrics
        map1 = sum(p == g for p, g in zip(preds, df["gold_type"])) / len(df)
        non_existent = sum(1 for p in preds if p not in MEDICAL_VALID_LETTERS) / len(df)

        return {
            "style": style,
            "MAP@1": map1,
            "% Non-existent": non_existent,
            "predictions": preds,
        }


def run_ontology_generation_validation():
    """Run ontology generation and validation experiment."""
    if not HAS_OWLREADY:
        print("owlready2 not available for ontology validation")
        return

    print("\n" + "=" * 60)
    print("ONTOLOGY GENERATION & VALIDATION")
    print("=" * 60)

    # Medical ontology terms and relations
    terms = [
        "AcuteMyeloidLeukemia",
        "Chemotherapy",
        "Symptom",
        "Diagnosis",
        "Treatment",
    ]
    relations = [
        ("AcuteMyeloidLeukemia", "has_treatment", "Chemotherapy"),
        ("AcuteMyeloidLeukemia", "diagnosed_by", "Diagnosis"),
        ("AcuteMyeloidLeukemia", "has_symptom", "Symptom"),
    ]

    if not HAS_OPENAI:
        print("OpenAI not available for ontology generation")
        return

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not set")
        return

    client = OpenAI(api_key=api_key)

    # Generate TTL
    prompt = f"""You are an ontology engineer.
Generate a valid OWL ontology in Turtle syntax.

Classes: {', '.join(terms)}
Relations: {'; '.join([f"{s} {p} {o}" for s, p, o in relations])}

Requirements:
- Plain Turtle syntax, no markdown
- Include @prefix for rdf, rdfs, owl, ex
- Define classes with rdf:type owl:Class
- Define object properties with domain/range
- Output ONLY valid TTL."""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        ttl = resp.choices[0].message.content
        ttl = re.sub(r"```.*?```", "", ttl, flags=re.S).strip()
        ttl = re.sub(r"```turtle\s*", "", ttl)
        ttl = re.sub(r"```\s*", "", ttl)

        print("\nGenerated TTL:")
        print("-" * 40)
        print(ttl[:500])
        print("-" * 40)

        # Validate with rdflib if available
        if HAS_RDFLIB:
            g = Graph()
            try:
                g.parse(data=ttl, format="turtle")
                print(f"✅ RDFLib parsed successfully: {len(g)} triples")
            except Exception as e:
                print(f"❌ RDFLib parse error: {e}")

    except Exception as e:
        print(f"Error generating ontology: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Term Typing Evaluation Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python term_typing_eval.py --task evaluate
  python term_typing_eval.py --task medical --style few-shot
  python term_typing_eval.py --data-file data/acute_myeloid_leukemia_2025.md
  python term_typing_eval.py --task ontology
        """,
    )

    parser.add_argument(
        "--task",
        type=str,
        default="evaluate",
        choices=["evaluate", "medical", "ontology", "all"],
        help="Task to run",
    )
    parser.add_argument(
        "--style",
        type=str,
        default="all",
        choices=["structured", "cot", "few-shot", "all"],
        help="Prompting style",
    )
    parser.add_argument("--data-file", type=str, default=None, help="Path to data file")
    parser.add_argument("--openai-key", type=str, default=None, help="OpenAI API key")
    parser.add_argument(
        "--max-samples", type=int, default=None, help="Maximum samples to evaluate"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Term Typing Evaluation Experiment")
    print("Medical Domain: Acute Myeloid Leukemia")
    print("=" * 70)

    # Set API key if provided
    if args.openai_key:
        os.environ["OPENAI_API_KEY"] = args.openai_key

    # Load data
    samples = MEDICAL_TERM_SAMPLES.copy()

    data_file = args.data_file
    if data_file is None:
        default_path = Path(__file__).parent / "data" / "acute_myeloid_leukemia_2025.md"
        if default_path.exists():
            data_file = str(default_path)

    if data_file:
        additional = load_additional_terms_from_file(data_file)
        samples.extend(additional[:10])  # Add up to 10 additional terms
        print(f"Added {len(additional[:10])} terms from {data_file}")

    df = pd.DataFrame(samples)
    if args.max_samples:
        df = df.head(args.max_samples)

    print(f"\nLoaded {len(df)} terms for evaluation")

    # Initialize evaluator
    evaluator = TermTypingEvaluator()

    # Run evaluation
    if args.task in ["evaluate", "medical", "all"]:
        styles = (
            ["structured", "cot", "few-shot"] if args.style == "all" else [args.style]
        )

        results = []
        for style in styles:
            result = evaluator.evaluate(df, style=style)
            results.append(result)

        # Summary
        print("\n" + "=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)

        summary_df = pd.DataFrame(
            [
                {
                    "Style": r["style"],
                    "MAP@1": f"{r['MAP@1']:.2%}",
                    "% Non-existent": f"{r['% Non-existent']:.2%}",
                }
                for r in results
            ]
        )

        print(summary_df.to_string(index=False))

    # Run ontology experiment
    if args.task in ["ontology", "all"]:
        run_ontology_generation_validation()

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(
        """
Key Findings:
- Few-shot prompting often outperforms zero-shot
- Chain-of-Thought can improve reasoning but may be verbose
- Structured prompting provides consistent output format
- Medical term typing requires domain-specific training
    """
    )


if __name__ == "__main__":
    main()
