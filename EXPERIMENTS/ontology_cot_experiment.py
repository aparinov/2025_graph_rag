#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ontology Experiment: Chain-of-Thought vs Structured Prompting

CLI version converted from Google Colab notebook.
Original notebook: https://colab.research.google.com/drive/17HUEPg20vlWXuw8_jh4yjs55E4kk8RUF

This experiment compares two prompting strategies for ontology learning:
a) Chain-of-Thought (CoT) - LLM reasons step-by-step before answering
b) Structured Prompting - strict format: term → predicted type

Metrics:
- MAP@1 (first answer accuracy)
- % "non-existent types" (types not in the ontology)

Uses medical domain data from acute_myeloid_leukemia_2025.md

Usage:
    python ontology_cot_experiment.py --task classify
    python ontology_cot_experiment.py --task generate-ontology
    python ontology_cot_experiment.py --task auto-fix
    python ontology_cot_experiment.py --data-file data/acute_myeloid_leukemia_2025.md
"""

import argparse
import json
import os
import re
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

import pandas as pd

# Optional imports
try:
    from openai import OpenAI

    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("Warning: openai not installed. OpenAI features disabled.")

try:
    from owlready2 import get_ontology, sync_reasoner_pellet

    HAS_OWLREADY = True
except ImportError:
    HAS_OWLREADY = False
    print("Warning: owlready2 not installed. Reasoner features disabled.")

try:
    from rdflib import Graph, Namespace, URIRef, Literal
    from rdflib.namespace import RDF, RDFS, OWL

    HAS_RDFLIB = True
except ImportError:
    HAS_RDFLIB = False
    print("Warning: rdflib not installed. RDF features disabled.")


# ============================================================================
# MEDICAL DOMAIN DATA (from acute_myeloid_leukemia_2025.md)
# ============================================================================

# Valid medical ontology classes for term typing
MEDICAL_CLASSES = {
    "Disease": [
        "leukemia",
        "lymphoma",
        "myelodysplastic",
        "cancer",
        "лейкоз",
        "лимфома",
    ],
    "Procedure": [
        "transplantation",
        "chemotherapy",
        "biopsy",
        "трансплантация",
        "химиотерапия",
        "биопсия",
    ],
    "DiagnosticMethod": [
        "immunophenotyping",
        "cytogenetic",
        "PCR",
        "FISH",
        "иммунофенотипирование",
        "цитогенетика",
        "ПЦР",
    ],
    "Symptom": ["fever", "bleeding", "лихорадка", "кровотечение", "геморрагический"],
    "Drug": ["cytarabine", "mercaptopurine", "цитарабин", "меркаптопурин"],
    "AnatomicalEntity": [
        "bone marrow",
        "CNS",
        "liver",
        "spleen",
        "костный мозг",
        "ЦНС",
        "печень",
        "селезенка",
    ],
    "GeneticMarker": ["FLT3", "NPM1", "RUNX1", "CBFB", "t(8;21)", "inv(16)"],
    "Condition": [
        "anemia",
        "neutropenia",
        "thrombocytopenia",
        "анемия",
        "нейтропения",
        "тромбоцитопения",
    ],
}

VALID_CLASSES = list(MEDICAL_CLASSES.keys())

# Sample medical terms for classification
MEDICAL_TERMS_FOR_CLASSIFICATION = [
    {"term": "острый миелоидный лейкоз", "gold_type": "Disease"},
    {"term": "химиотерапия", "gold_type": "Procedure"},
    {"term": "иммунофенотипирование", "gold_type": "DiagnosticMethod"},
    {"term": "лихорадка", "gold_type": "Symptom"},
    {"term": "цитарабин", "gold_type": "Drug"},
    {"term": "костный мозг", "gold_type": "AnatomicalEntity"},
    {"term": "FLT3", "gold_type": "GeneticMarker"},
    {"term": "нейтропения", "gold_type": "Condition"},
    {
        "term": "трансплантация гемопоэтических стволовых клеток",
        "gold_type": "Procedure",
    },
    {"term": "миелодиспластический синдром", "gold_type": "Disease"},
    {"term": "RUNX1-RUNX1T1", "gold_type": "GeneticMarker"},
    {"term": "спинномозговая пункция", "gold_type": "Procedure"},
]

# Ontology terms and relations for generation
ONTOLOGY_TERMS = [
    "AcuteMyeloidLeukemia",
    "ChemotherapyDrug",
    "DiagnosticTest",
    "GeneticMarker",
    "Symptom",
    "TreatmentProtocol",
    "Patient",
    "BoneMarrow",
    "BloodCell",
    "Remission",
]

ONTOLOGY_RELATIONS = [
    ("AcuteMyeloidLeukemia", "treated_with", "ChemotherapyDrug"),
    ("AcuteMyeloidLeukemia", "diagnosed_by", "DiagnosticTest"),
    ("AcuteMyeloidLeukemia", "has_marker", "GeneticMarker"),
    ("AcuteMyeloidLeukemia", "has_symptom", "Symptom"),
    ("Patient", "has_disease", "AcuteMyeloidLeukemia"),
    ("Patient", "undergoes", "TreatmentProtocol"),
    ("TreatmentProtocol", "includes", "ChemotherapyDrug"),
    ("BoneMarrow", "contains", "BloodCell"),
    ("AcuteMyeloidLeukemia", "affects", "BoneMarrow"),
    ("TreatmentProtocol", "leads_to", "Remission"),
]


def load_medical_data_from_file(filepath: str) -> pd.DataFrame:
    """Load medical data from markdown file and extract terms."""
    df = pd.DataFrame(MEDICAL_TERMS_FOR_CLASSIFICATION)

    if filepath and Path(filepath).exists():
        print(f"Loading additional data from: {filepath}")
        content = Path(filepath).read_text(encoding="utf-8", errors="ignore")

        # Extract abbreviations as additional terms
        abbreviations = re.findall(r"\b([А-ЯA-Z]{2,6})\s+[–—-]\s+([^,\n]+)", content)
        additional_terms = []

        for abbr, definition in abbreviations[:15]:
            # Guess type based on definition content
            gold_type = "Abbreviation"
            def_lower = definition.lower()
            if any(
                kw in def_lower for kw in ["лейкоз", "синдром", "болезнь", "leukemia"]
            ):
                gold_type = "Disease"
            elif any(
                kw in def_lower for kw in ["терапия", "трансплантация", "therapy"]
            ):
                gold_type = "Procedure"
            elif any(kw in def_lower for kw in ["исследование", "test", "метод"]):
                gold_type = "DiagnosticMethod"

            additional_terms.append({"term": abbr, "gold_type": gold_type})

        if additional_terms:
            df = pd.concat([df, pd.DataFrame(additional_terms)], ignore_index=True)

    return df


class OntologyExperiment:
    """Main class for ontology experiments."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = None
        if HAS_OPENAI and self.api_key:
            self.client = OpenAI(api_key=self.api_key)

    def predict_cot(self, term: str, context: str = None) -> str:
        """Predict type using Chain-of-Thought prompting."""
        if not self.client:
            return "Unknown"

        prompt = f"""Classify the medical/biomedical term '{term}' into one of these classes: {', '.join(VALID_CLASSES)}.

Think step-by-step:
1. What does this term refer to?
2. What category does it belong to?
3. What is the final classification?

{f"Context: {context}" if context else ""}

Think briefly, then return ONLY the class name at the end."""

        try:
            resp = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            result = resp.choices[0].message.content.strip()
            # Extract the final class from response
            for cls in VALID_CLASSES:
                if cls.lower() in result.lower():
                    return cls
            return result.split()[-1] if result else "Unknown"
        except Exception as e:
            print(f"Error in CoT prediction: {e}")
            return "Unknown"

    def predict_structured(self, term: str, context: str = None) -> str:
        """Predict type using Structured prompting."""
        if not self.client:
            return "Unknown"

        prompt = f"""Classify the medical term '{term}' into exactly ONE of these classes: {', '.join(VALID_CLASSES)}.
{f"Context: {context}" if context else ""}
Return ONLY the class name, nothing else."""

        try:
            resp = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            result = resp.choices[0].message.content.strip()
            # Clean up response
            for cls in VALID_CLASSES:
                if cls.lower() in result.lower():
                    return cls
            return result
        except Exception as e:
            print(f"Error in structured prediction: {e}")
            return "Unknown"

    def evaluate_classification(
        self, df: pd.DataFrame, style: str = "structured"
    ) -> Dict[str, Any]:
        """Evaluate classification with given prompting style."""
        preds = []

        predict_fn = self.predict_cot if style == "cot" else self.predict_structured

        print(f"\nEvaluating with {style.upper()} prompting...")
        for _, row in df.iterrows():
            pred = predict_fn(row["term"])
            preds.append(pred)
            print(
                f"  Term: {row['term'][:40]:40s} | Gold: {row['gold_type']:15s} | Pred: {pred}"
            )

        # Calculate metrics
        df_eval = df.copy()
        df_eval["pred"] = preds

        map1 = sum(
            p.lower() == g.lower() for p, g in zip(preds, df["gold_type"])
        ) / len(df)
        non_existent = sum(1 for p in preds if p not in VALID_CLASSES) / len(df)

        return {
            "style": style,
            "MAP@1": map1,
            "% Non-existent": non_existent,
            "predictions": preds,
        }

    def generate_ontology_ttl(
        self, terms: List[str], relations: List[Tuple[str, str, str]]
    ) -> str:
        """Generate OWL ontology in Turtle format using LLM."""
        if not self.client:
            print("OpenAI client not available")
            return ""

        relations_str = "; ".join([f"{s} {p} {o}" for s, p, o in relations])

        prompt = f"""You are an ontology engineer.
Generate a valid OWL ontology in Turtle syntax based on the following classes and relations.

Classes: {', '.join(terms)}
Relations: {relations_str}

Requirements:
- Use only plain Turtle syntax, no Markdown formatting, no code fences.
- Include @prefix declarations for rdf, rdfs, owl, and ex (<http://example.org/>).
- Define each class with rdf:type owl:Class.
- Define each object property with rdfs:domain and rdfs:range.
- Ensure the ontology is logically consistent.
- Output ONLY valid TTL."""

        try:
            resp = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            ttl_content = resp.choices[0].message.content
            # Remove markdown fences if present
            ttl_content = re.sub(r"```.*?```", "", ttl_content, flags=re.S)
            ttl_content = re.sub(r"```turtle\s*", "", ttl_content)
            ttl_content = re.sub(r"```\s*", "", ttl_content)
            return ttl_content.strip()
        except Exception as e:
            print(f"Error generating ontology: {e}")
            return ""

    def validate_ontology(self, ttl_content: str) -> Tuple[bool, str, List]:
        """Validate ontology using owlready2 reasoner."""
        if not HAS_OWLREADY:
            return False, "owlready2 not installed", []

        try:
            # Save to temp file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".ttl", delete=False, encoding="utf-8"
            ) as f:
                f.write(ttl_content)
                temp_path = f.name

            # First try to parse with rdflib
            if HAS_RDFLIB:
                g = Graph()
                g.parse(temp_path, format="turtle")

                # Convert to RDF/XML for owlready2
                xml_path = temp_path.replace(".ttl", ".owl")
                g.serialize(destination=xml_path, format="xml")
            else:
                xml_path = temp_path

            # Load with owlready2 and run reasoner
            onto = get_ontology(f"file://{xml_path}").load()
            with onto:
                sync_reasoner_pellet(
                    infer_property_values=True, infer_data_property_values=True
                )

            # Check for inconsistent classes
            inconsistent = [
                c
                for c in onto.classes()
                if c.equivalent_to and "Nothing" in str(c.equivalent_to)
            ]

            # Cleanup
            os.unlink(temp_path)
            if xml_path != temp_path and os.path.exists(xml_path):
                os.unlink(xml_path)

            return (
                True,
                f"Valid ontology. Inconsistent classes: {len(inconsistent)}",
                inconsistent,
            )

        except Exception as e:
            return False, str(e), []

    def auto_fix_ontology(
        self, ttl_content: str, error_msg: str, max_attempts: int = 3
    ) -> str:
        """Attempt to auto-fix ontology using LLM."""
        if not self.client:
            return ttl_content

        current_ttl = ttl_content

        for attempt in range(max_attempts):
            print(f"\nAttempt {attempt + 1} to fix ontology...")

            fix_prompt = f"""The following Turtle ontology failed to parse or had logical issues:

{current_ttl}

Error: {error_msg}

Please FIX ALL issues. Output ONLY valid Turtle (no markdown fences)."""

            try:
                resp = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": fix_prompt}],
                    temperature=0,
                )
                current_ttl = resp.choices[0].message.content
                current_ttl = re.sub(r"```.*?```", "", current_ttl, flags=re.S).strip()
                current_ttl = re.sub(r"```turtle\s*", "", current_ttl)
                current_ttl = re.sub(r"```\s*", "", current_ttl)

                # Try to validate
                ok, msg, _ = self.validate_ontology(current_ttl)
                if ok:
                    print(f"✅ Fixed successfully on attempt {attempt + 1}")
                    return current_ttl
                else:
                    error_msg = msg

            except Exception as e:
                error_msg = str(e)
                print(f"Fix attempt failed: {e}")

        print("❌ Failed to fix ontology after all attempts")
        return current_ttl


def run_classification_experiment(
    experiment: OntologyExperiment, df: pd.DataFrame
) -> pd.DataFrame:
    """Run classification experiment comparing CoT vs Structured prompting."""
    results = []

    print("\n" + "=" * 70)
    print("CLASSIFICATION EXPERIMENT: Chain-of-Thought vs Structured Prompting")
    print("=" * 70)

    for style in ["cot", "structured"]:
        result = experiment.evaluate_classification(df, style=style)
        results.append(result)
        print(f"\n{style.upper()} Results:")
        print(f"  MAP@1: {result['MAP@1']:.2%}")
        print(f"  % Non-existent: {result['% Non-existent']:.2%}")

    return pd.DataFrame(
        [
            {
                "Style": r["style"],
                "MAP@1": f"{r['MAP@1']:.2%}",
                "% Non-existent": f"{r['% Non-existent']:.2%}",
            }
            for r in results
        ]
    )


def run_ontology_generation(
    experiment: OntologyExperiment, output_file: str = "generated_ontology.ttl"
):
    """Run ontology generation experiment."""
    print("\n" + "=" * 70)
    print("ONTOLOGY GENERATION EXPERIMENT")
    print("=" * 70)

    print(
        f"\nGenerating ontology with {len(ONTOLOGY_TERMS)} classes and {len(ONTOLOGY_RELATIONS)} relations..."
    )

    ttl = experiment.generate_ontology_ttl(ONTOLOGY_TERMS, ONTOLOGY_RELATIONS)

    if ttl:
        print(f"\nGenerated TTL ({len(ttl)} chars):")
        print("-" * 40)
        print(ttl[:500] + "..." if len(ttl) > 500 else ttl)
        print("-" * 40)

        # Validate
        ok, msg, inconsistent = experiment.validate_ontology(ttl)
        print(f"\nValidation: {'✅ PASSED' if ok else '❌ FAILED'}")
        print(f"Message: {msg}")

        if not ok:
            print("\nAttempting auto-fix...")
            ttl = experiment.auto_fix_ontology(ttl, msg)
            ok, msg, _ = experiment.validate_ontology(ttl)
            print(f"After fix: {'✅ PASSED' if ok else '❌ FAILED'}")

        # Save
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(ttl)
        print(f"\n✅ Ontology saved to {output_file}")

        return ttl

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Ontology Experiment: CoT vs Structured Prompting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ontology_cot_experiment.py --task classify
  python ontology_cot_experiment.py --task generate-ontology --output ontology.ttl
  python ontology_cot_experiment.py --task all --data-file data/acute_myeloid_leukemia_2025.md
        """,
    )

    parser.add_argument(
        "--task",
        type=str,
        default="all",
        choices=["all", "classify", "generate-ontology", "auto-fix"],
        help="Task to run",
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default=None,
        help="Path to data file (e.g., acute_myeloid_leukemia_2025.md)",
    )
    parser.add_argument(
        "--openai-key",
        type=str,
        default=None,
        help="OpenAI API key (or set OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="generated_ontology.ttl",
        help="Output file for generated ontology",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples for classification (default: all)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Ontology Experiment: Chain-of-Thought vs Structured Prompting")
    print("Medical Domain: Acute Myeloid Leukemia")
    print("=" * 70)

    # Check for API key
    api_key = args.openai_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n⚠️  Warning: No OpenAI API key provided.")
        print("Set OPENAI_API_KEY environment variable or use --openai-key")
        print("Running in demo mode with limited functionality.\n")

    # Initialize experiment
    experiment = OntologyExperiment(api_key=api_key)

    # Load data
    data_file = args.data_file
    if data_file is None:
        default_path = Path(__file__).parent / "data" / "acute_myeloid_leukemia_2025.md"
        if default_path.exists():
            data_file = str(default_path)

    df = load_medical_data_from_file(data_file)
    if args.max_samples:
        df = df.head(args.max_samples)

    print(f"\nLoaded {len(df)} terms for classification")

    # Run tasks
    if args.task in ["all", "classify"]:
        results_df = run_classification_experiment(experiment, df)
        print("\n" + "=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)
        print(results_df.to_string(index=False))

    if args.task in ["all", "generate-ontology", "auto-fix"]:
        run_ontology_generation(experiment, args.output)

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(
        """
Key Findings:
- Chain-of-Thought (CoT) prompting encourages step-by-step reasoning
- Structured prompting provides more consistent output format
- Both approaches can produce non-existent types without proper constraints
- Auto-fix mechanism can improve ontology consistency
    """
    )


if __name__ == "__main__":
    main()
