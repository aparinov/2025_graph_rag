# -*- coding: utf-8 -*-

from pyvis.network import Network
import networkx as nx
import html as html_lib
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json

try:
    import json_repair
except ImportError:
    raise ImportError(
        "The 'json_repair' library is required. Please install it with 'pip install json-repair'"
    )


def strip_code_fences(text: str) -> str:
    """Remove markdown code fences from text"""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if len(lines) > 1:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


def extract_relations_from_chunks(
    chunks: List[Dict[str, Any]],
    llm: ChatOpenAI,
    relation_prompt: ChatPromptTemplate
) -> List[Dict]:
    """Extract relations from retrieved chunks"""

    all_relations = []

    # Combine top chunks for context
    chunk_texts = [c["text"] for c in chunks[:3]]
    combined_text = "\n\n".join(chunk_texts)

    # Collect all entities from chunks
    all_entities = {}
    for chunk_data in chunks:
        if "entities" in chunk_data and chunk_data["entities"]:
            for ent in chunk_data["entities"]:
                ent_key = f"{ent['name']}::{ent['type']}"
                if ent_key not in all_entities:
                    all_entities[ent_key] = ent

    # Only extract relations if we have at least 2 entities
    if len(all_entities) < 2:
        return []

    # Extract relations using LLM
    try:
        chain = relation_prompt | llm | StrOutputParser()
        entities_list = list(all_entities.values())

        response = chain.invoke({
            "text": combined_text[:2000],
            "entities": json.dumps(entities_list, ensure_ascii=False)
        })

        parsed = json_repair.loads(strip_code_fences(response))

        if isinstance(parsed, list):
            all_relations.extend(parsed)

    except Exception as e:
        print(f"[WARNING] Failed to extract relations: {e}")

    return all_relations


def build_graph_html(entities: List[Dict], relations: List[Dict]) -> str:
    """Build interactive PyVis graph HTML"""

    if not entities:
        return "<div style='padding: 20px; text-align: center; color: #666;'>Граф будет отображен после того, как вы зададите вопрос</div>"

    # Create NetworkX graph
    G = nx.DiGraph()

    # Color mapping for entity types
    color_map = {
        "Препарат": "#E74C3C",
        "ДействующееВещество": "#C0392B",
        "ФармакологическаяГруппа": "#E67E22",
        "ЛекарственнаяФорма": "#F39C12",
        "Дозировка": "#3498DB",
        "Показание": "#2ECC71",
        "Противопоказание": "#E67E22",
        "Побочноедействие": "#FF8C42",
        "ДиагностическийМетод": "#9B59B6",
        "МетодЛечения": "#1ABC9C",
        "МедицинскаяПроцедура": "#16A085",
        "Заболевание": "#F39C12",
        "Симптом": "#16A085",
        "Синдром": "#D35400",
        "СтадияЗаболевания": "#8E44AD",
        "КодМКБ": "#2C3E50",
        "ЛабораторныйПоказатель": "#34495E",
        "Специалист": "#7F8C8D",
        "МедицинскаяОрганизация": "#95A5A6",
        "Возрастнаякатегория": "#BDC3C7",
        "ФизиологическоеСостояние": "#E8DAEF",
    }

    # Add nodes
    for entity in entities:
        entity_name = entity.get("name", "Unknown")
        entity_type = entity.get("type", "Unknown")

        # Build title with properties if available
        title_parts = [f"<b>{entity_type}</b>: {entity_name}"]
        if "properties" in entity and entity["properties"]:
            for key, value in entity["properties"].items():
                title_parts.append(f"{key}: {value}")
        title = "<br>".join(title_parts)

        G.add_node(
            entity_name,
            label=entity_name,
            title=title,
            color=color_map.get(entity_type, "#95A5A6"),
            size=25
        )

    # Add edges
    for rel in relations:
        source = rel.get("source", "")
        target = rel.get("target", "")
        rel_type = rel.get("type", "связан_с")

        if source in G.nodes and target in G.nodes:
            G.add_edge(
                source,
                target,
                label=rel_type,
                title=rel_type,
                arrows="to"
            )

    # If no edges, add some based on entity co-occurrence
    if G.number_of_edges() == 0 and G.number_of_nodes() > 1:
        nodes = list(G.nodes())
        # Connect entities that appear in the same context
        for i in range(min(5, len(nodes) - 1)):
            G.add_edge(
                nodes[i],
                nodes[i + 1],
                label="связан_с",
                title="связан_с",
                arrows="to"
            )

    # Create PyVis network
    net = Network(
        height="600px",
        width="100%",
        directed=True,
        notebook=False,
        cdn_resources="in_line"
    )

    # Configure physics for better layout
    net.set_options("""
    {
        "physics": {
            "barnesHut": {
                "gravitationalConstant": -8000,
                "centralGravity": 0.3,
                "springLength": 150,
                "springConstant": 0.04,
                "damping": 0.09,
                "avoidOverlap": 0.1
            },
            "maxVelocity": 50,
            "minVelocity": 0.1,
            "stabilization": {
                "iterations": 100,
                "updateInterval": 25
            }
        },
        "nodes": {
            "font": {
                "size": 14,
                "face": "Arial"
            },
            "borderWidth": 2,
            "borderWidthSelected": 3
        },
        "edges": {
            "font": {
                "size": 12,
                "align": "middle"
            },
            "arrows": {
                "to": {
                    "enabled": true,
                    "scaleFactor": 0.5
                }
            },
            "smooth": {
                "type": "continuous"
            }
        },
        "interaction": {
            "hover": true,
            "tooltipDelay": 100,
            "navigationButtons": true,
            "keyboard": {
                "enabled": true
            }
        }
    }
    """)

    # Add nodes and edges from NetworkX
    net.from_nx(G)

    # Generate HTML
    raw_html = net.generate_html()
    escaped = html_lib.escape(raw_html, quote=True)
    return (
        "<iframe"
        " style=\"width: 100%; height: 600px; border: 0;\""
        " srcdoc=\""
        + escaped +
        "\"></iframe>"
    )
