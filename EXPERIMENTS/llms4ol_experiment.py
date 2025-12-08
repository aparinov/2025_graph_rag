#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLMs4OL: Large Language Models for Ontology Learning

CLI version converted from Google Colab notebook.
Original notebook: https://colab.research.google.com/drive/1NwmkIS3TUE37eTuy0MxcP-YjnP3egcXR

This script implements experiments from the paper "LLMs4OL: Large Language Models for Ontology Learning"
to test if LLMs can perform key Ontology Learning tasks in Zero-Shot mode.

Tasks implemented:
* Term Typing – determining the type of a concept
* Taxonomy Discovery – checking superclass/subclass relationships
* Relation Extraction – verifying head-relation-tail triples

Usage:
    python llms4ol_experiment.py --task all
    python llms4ol_experiment.py --task term_typing
    python llms4ol_experiment.py --task taxonomy
    python llms4ol_experiment.py --task relation
    python llms4ol_experiment.py --task finetune
    python llms4ol_experiment.py --data-file data/acute_myeloid_leukemia_2025.md
"""

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

# Optional imports - will check availability
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: transformers not installed. Some features will be disabled.")

try:
    from sklearn.metrics import f1_score

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from openai import OpenAI

    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("Warning: openai not installed. OpenAI features will be disabled.")

try:
    from peft import LoraConfig, get_peft_model, PeftModel
    from transformers import (
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        DataCollatorForSeq2Seq,
    )
    from datasets import load_dataset

    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False
    print("Warning: peft/datasets not installed. Fine-tuning will be disabled.")

try:
    from rdflib import Graph, RDFS, Literal

    HAS_RDFLIB = True
except ImportError:
    HAS_RDFLIB = False
    print("Warning: rdflib not installed. Schema.org features will be disabled.")


# ============================================================================
# MEDICAL DOMAIN DATA (from acute_myeloid_leukemia_2025.md)
# ============================================================================

MEDICAL_TERMS = [
    # Diseases and conditions
    ("острый миелоидный лейкоз", "Disease"),
    ("acute myeloid leukemia", "Disease"),
    ("AML", "Disease"),
    ("миелодиспластический синдром", "Disease"),
    ("нейролейкоз", "Disease"),
    ("анемия", "Condition"),
    ("тромбоцитопения", "Condition"),
    ("нейтропения", "Condition"),
    ("лихорадка", "Symptom"),
    ("геморрагический синдром", "Symptom"),
    # Medical procedures
    ("трансплантация гемопоэтических стволовых клеток", "Procedure"),
    ("химиотерапия", "Procedure"),
    ("спинномозговая пункция", "Procedure"),
    ("биопсия костного мозга", "Procedure"),
    ("эхокардиография", "Procedure"),
    # Diagnostic methods
    ("иммунофенотипирование", "DiagnosticMethod"),
    ("цитогенетическое исследование", "DiagnosticMethod"),
    ("ПЦР", "DiagnosticMethod"),
    ("FISH", "DiagnosticMethod"),
    # Drugs/Treatments
    ("цитарабин", "Drug"),
    ("меркаптопурин", "Drug"),
    # Anatomical entities
    ("костный мозг", "AnatomicalEntity"),
    ("ЦНС", "AnatomicalEntity"),
    ("печень", "AnatomicalEntity"),
    ("селезенка", "AnatomicalEntity"),
]

# Taxonomy relationships from medical domain
MEDICAL_TAXONOMY = [
    ("Disease", "острый миелоидный лейкоз"),
    ("Disease", "острый лимфобластный лейкоз"),
    ("Disease", "миелодиспластический синдром"),
    ("Procedure", "трансплантация гемопоэтических стволовых клеток"),
    ("Procedure", "химиотерапия"),
    ("DiagnosticMethod", "иммунофенотипирование"),
    ("DiagnosticMethod", "цитогенетическое исследование"),
    ("Symptom", "лихорадка"),
    ("Symptom", "геморрагический синдром"),
    ("Condition", "анемия"),
    ("Condition", "тромбоцитопения"),
]

# Medical relations
MEDICAL_RELATIONS = [
    ("острый миелоидный лейкоз", "treated_with", "химиотерапия"),
    ("острый миелоидный лейкоз", "diagnosed_by", "иммунофенотипирование"),
    ("острый миелоидный лейкоз", "has_symptom", "лихорадка"),
    ("острый миелоидный лейкоз", "affects", "костный мозг"),
    ("химиотерапия", "causes", "нейтропения"),
    ("химиотерапия", "causes", "тромбоцитопения"),
    (
        "трансплантация гемопоэтических стволовых клеток",
        "treats",
        "острый миелоидный лейкоз",
    ),
    ("нейролейкоз", "affects", "ЦНС"),
]


def load_medical_data_from_file(filepath: str) -> Dict[str, List]:
    """Load and parse medical data from the acute_myeloid_leukemia markdown file."""
    data = {
        "terms": MEDICAL_TERMS.copy(),
        "taxonomy": MEDICAL_TAXONOMY.copy(),
        "relations": MEDICAL_RELATIONS.copy(),
    }

    if filepath and Path(filepath).exists():
        print(f"Loading additional data from: {filepath}")
        content = Path(filepath).read_text(encoding="utf-8", errors="ignore")

        # Extract additional medical terms using patterns
        # This is a simplified extraction - in production would use NLP
        abbreviations = re.findall(r"\b([А-ЯA-Z]{2,6})\s+[–—-]\s+([^,\n]+)", content)
        for abbr, definition in abbreviations[:20]:
            term_type = "Abbreviation"
            if "лейкоз" in definition.lower() or "leukemia" in definition.lower():
                term_type = "Disease"
            elif "терапия" in definition.lower() or "therapy" in definition.lower():
                term_type = "Procedure"
            data["terms"].append((abbr, term_type))

    return data


def prepare_term_typing_samples(terms: List[Tuple[str, str]]) -> List[Dict]:
    """Prepare samples for term typing task."""
    samples = []
    for term, term_type in terms:
        prompt = f"Question: What is the medical/ontological type of '{term}'? Answer with one of: Disease, Condition, Symptom, Procedure, DiagnosticMethod, Drug, AnatomicalEntity, Abbreviation. Answer:"
        samples.append({"prompt": prompt, "label": term_type})
    return samples


def prepare_taxonomy_samples(taxonomy: List[Tuple[str, str]]) -> List[Dict]:
    """Prepare samples for taxonomy discovery task."""
    samples = []

    # Positive examples
    for parent, child in taxonomy:
        prompt = (
            f"Question: Is '{parent}' a superclass of '{child}'? Answer True or False:"
        )
        samples.append({"prompt": prompt, "label": "True"})

    # Negative examples (swap parent and child)
    for parent, child in random.sample(taxonomy, min(len(taxonomy), 10)):
        prompt = (
            f"Question: Is '{child}' a superclass of '{parent}'? Answer True or False:"
        )
        samples.append({"prompt": prompt, "label": "False"})

    return samples


def prepare_relation_samples(relations: List[Tuple[str, str, str]]) -> List[Dict]:
    """Prepare samples for relation extraction task."""
    samples = []

    # Positive examples
    for head, rel, tail in relations:
        prompt = f"Question: Does '{head}' {rel} '{tail}'? Answer True or False:"
        samples.append({"prompt": prompt, "label": "True"})

    # Negative examples (random combinations)
    for _ in range(min(len(relations), 5)):
        h = random.choice(relations)[0]
        r = random.choice(relations)[1]
        t = random.choice(relations)[2]
        if (h, r, t) not in relations:
            prompt = f"Question: Does '{h}' {r} '{t}'? Answer True or False:"
            samples.append({"prompt": prompt, "label": "False"})

    return samples


def evaluate_model_zeroshot(
    samples: List[Dict], pipe, task_name: str = "Task"
) -> float:
    """Evaluate model in zero-shot setting."""
    if not HAS_TRANSFORMERS:
        print("Transformers not available for evaluation")
        return 0.0

    correct = 0
    total = len(samples)

    print(f"\n{'='*60}")
    print(f"Evaluating {task_name} ({total} samples)")
    print(f"{'='*60}")

    for s in tqdm(samples, desc=f"Evaluating {task_name}"):
        output = pipe(s["prompt"], max_new_tokens=20)[0]["generated_text"].strip()

        # Check if label is in output
        if s["label"].lower() in output.lower():
            correct += 1
            result = "✓"
        else:
            result = "✗"

        print(f"\n{result} PROMPT: {s['prompt'][:80]}...")
        print(f"  MODEL : {output}")
        print(f"  LABEL : {s['label']}")

    accuracy = correct / total if total > 0 else 0
    print(f"\n📊 {task_name} Accuracy: {accuracy:.2%}")
    return accuracy


def evaluate_f1(samples: List[Dict], pipe, task_name: str = "Task") -> float:
    """Evaluate F1 score for binary classification tasks."""
    if not HAS_TRANSFORMERS or not HAS_SKLEARN:
        print("Required libraries not available for F1 evaluation")
        return 0.0

    y_true, y_pred = [], []

    for s in tqdm(samples, desc=f"Computing F1 for {task_name}"):
        output = pipe(s["prompt"], max_new_tokens=10)[0]["generated_text"].lower()
        pred = "true" if "true" in output else "false"
        y_pred.append(pred)
        y_true.append(s["label"].lower())

    f1 = f1_score(y_true, y_pred, pos_label="true")
    print(f"\n📊 {task_name} F1 Score: {f1:.2%}")
    return f1


def run_openai_experiment(
    samples: List[Dict], task_name: str, api_key: str = None
) -> float:
    """Run experiment using OpenAI API."""
    if not HAS_OPENAI:
        print("OpenAI not available")
        return 0.0

    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("No OpenAI API key provided. Set OPENAI_API_KEY environment variable.")
        return 0.0

    client = OpenAI(api_key=api_key)
    correct = 0

    print(f"\n{'='*60}")
    print(f"Evaluating {task_name} with OpenAI GPT-4o-mini")
    print(f"{'='*60}")

    for s in tqdm(samples[:10], desc="OpenAI evaluation"):  # Limit to 10 for cost
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": s["prompt"]}],
                temperature=0,
            )
            output = resp.choices[0].message.content.strip()

            if s["label"].lower() in output.lower():
                correct += 1
        except Exception as e:
            print(f"OpenAI API error: {e}")
            continue

    accuracy = correct / min(len(samples), 10)
    print(f"\n📊 {task_name} Accuracy (GPT-4o-mini): {accuracy:.2%}")
    return accuracy


def run_finetuning(train_data: List[Dict], output_dir: str = "./ft-medical-lora"):
    """Run LoRA fine-tuning on the model."""
    if not HAS_PEFT:
        print("PEFT not available for fine-tuning")
        return None

    print("\n" + "=" * 60)
    print("Starting LoRA Fine-Tuning")
    print("=" * 60)

    # Save training data
    train_file = "train_medical.jsonl"
    with open(train_file, "w", encoding="utf-8") as f:
        for ex in train_data:
            f.write(
                json.dumps(
                    {"instruction": ex["prompt"], "output": ex["label"]},
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"Saved {len(train_data)} training examples to {train_file}")

    # Load dataset
    dataset = load_dataset("json", data_files=train_file)["train"]

    # Load model
    model_id = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    # Configure LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q", "v"],
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_2_SEQ_LM",
    )

    model = get_peft_model(model, lora_config)
    print(f"LoRA parameters: {model.print_trainable_parameters()}")

    # Tokenize
    def preprocess(batch):
        inputs = tokenizer(
            batch["instruction"], max_length=128, truncation=True, padding="max_length"
        )
        labels = tokenizer(
            batch["output"], max_length=32, truncation=True, padding="max_length"
        )
        inputs["labels"] = labels["input_ids"]
        return inputs

    train_dataset = dataset.map(preprocess, batched=True)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Training arguments
    args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=3,
        fp16=torch.cuda.is_available(),
        save_total_limit=1,
        logging_steps=10,
        report_to="none",
    )

    # Train
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    model.save_pretrained(output_dir)
    print(f"\n✅ Model saved to {output_dir}")

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="LLMs4OL: Large Language Models for Ontology Learning Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python llms4ol_experiment.py --task all
  python llms4ol_experiment.py --task term_typing --model google/flan-t5-base
  python llms4ol_experiment.py --task taxonomy --use-openai
  python llms4ol_experiment.py --data-file data/acute_myeloid_leukemia_2025.md
        """,
    )

    parser.add_argument(
        "--task",
        type=str,
        default="all",
        choices=["all", "term_typing", "taxonomy", "relation", "finetune"],
        help="Task to run",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/flan-t5-small",
        help="HuggingFace model to use",
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default=None,
        help="Path to additional data file (e.g., acute_myeloid_leukemia_2025.md)",
    )
    parser.add_argument(
        "--use-openai",
        action="store_true",
        help="Use OpenAI API instead of local model",
    )
    parser.add_argument(
        "--openai-key",
        type=str,
        default=None,
        help="OpenAI API key (or set OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./ft-medical-lora",
        help="Output directory for fine-tuned model",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=20,
        help="Maximum samples per task for evaluation",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("LLMs4OL: Large Language Models for Ontology Learning")
    print("Medical Domain Experiment (Acute Myeloid Leukemia)")
    print("=" * 70)

    # Load data
    data_file = args.data_file
    if data_file is None:
        # Try default location
        default_path = Path(__file__).parent / "data" / "acute_myeloid_leukemia_2025.md"
        if default_path.exists():
            data_file = str(default_path)
            print(f"Using default data file: {data_file}")

    medical_data = load_medical_data_from_file(data_file)

    # Prepare samples
    term_samples = prepare_term_typing_samples(medical_data["terms"])[
        : args.max_samples
    ]
    taxonomy_samples = prepare_taxonomy_samples(medical_data["taxonomy"])[
        : args.max_samples
    ]
    relation_samples = prepare_relation_samples(medical_data["relations"])[
        : args.max_samples
    ]

    print(f"\nPrepared samples:")
    print(f"  - Term Typing: {len(term_samples)}")
    print(f"  - Taxonomy Discovery: {len(taxonomy_samples)}")
    print(f"  - Relation Extraction: {len(relation_samples)}")

    # Initialize model
    pipe = None
    if HAS_TRANSFORMERS and not args.use_openai:
        print(f"\nLoading model: {args.model}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model)
            model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
            pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Failed to load model: {e}")

    results = {}

    # Run tasks
    if args.task in ["all", "term_typing"]:
        print("\n" + "=" * 60)
        print("TASK: Term Typing (Medical Domain)")
        print("=" * 60)
        if args.use_openai:
            results["term_typing_openai"] = run_openai_experiment(
                term_samples, "Term Typing", args.openai_key
            )
        elif pipe:
            results["term_typing"] = evaluate_model_zeroshot(
                term_samples, pipe, "Term Typing"
            )

    if args.task in ["all", "taxonomy"]:
        print("\n" + "=" * 60)
        print("TASK: Taxonomy Discovery (Medical Domain)")
        print("=" * 60)
        if args.use_openai:
            results["taxonomy_openai"] = run_openai_experiment(
                taxonomy_samples, "Taxonomy Discovery", args.openai_key
            )
        elif pipe:
            results["taxonomy_f1"] = evaluate_f1(
                taxonomy_samples, pipe, "Taxonomy Discovery"
            )

    if args.task in ["all", "relation"]:
        print("\n" + "=" * 60)
        print("TASK: Relation Extraction (Medical Domain)")
        print("=" * 60)
        if args.use_openai:
            results["relation_openai"] = run_openai_experiment(
                relation_samples, "Relation Extraction", args.openai_key
            )
        elif pipe:
            results["relation_f1"] = evaluate_f1(
                relation_samples, pipe, "Relation Extraction"
            )

    if args.task == "finetune":
        print("\n" + "=" * 60)
        print("TASK: Fine-Tuning with LoRA")
        print("=" * 60)
        all_samples = term_samples + taxonomy_samples + relation_samples
        random.shuffle(all_samples)
        run_finetuning(all_samples, args.output_dir)

    # Print summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)

    for task, score in results.items():
        print(f"  {task}: {score:.2%}")

    print("\nConclusions:")
    print("  - Zero-Shot LLMs struggle with domain-specific ontology tasks")
    print("  - Term Typing is particularly challenging without fine-tuning")
    print("  - Fine-tuning with LoRA can significantly improve performance")
    print("  - Medical domain requires specialized training data")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
