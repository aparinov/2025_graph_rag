# -*- coding: utf-8 -*-

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import httpx
import gradio as gr
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredPDFLoader, UnstructuredMarkdownLoader
from langchain_openai import ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from qdrant_manager import QdrantManager
from graph_builder import build_graph_html, extract_relations_from_chunks

try:
    import json_repair
except ImportError:
    raise ImportError(
        "The 'json_repair' library is required. Please install it with 'pip install json-repair'"
    )


load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
PROXY_URL = os.getenv("PROXY_URL") or os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
MAX_CHUNK_CHARS = int(os.getenv("MAX_CHUNK_CHARS", "2000"))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))

OPENAI_INGEST_MODEL = os.getenv("OPENAI_INGEST_MODEL", "gpt-4o-mini")
OPENAI_QA_MODEL = os.getenv("OPENAI_QA_MODEL", "gpt-4o-mini")


qdrant = QdrantManager(url=QDRANT_URL)


_LABEL_SAFE_RE = re.compile(r"[^0-9A-Za-z_\u0400-\u04FF]")


def build_http_client() -> httpx.Client:
    verify = os.getenv("SSL_CERT_FILE") or True
    transport = httpx.HTTPTransport(retries=3)
    if PROXY_URL:
        try:
            return httpx.Client(
                proxies=PROXY_URL, timeout=60, verify=verify, transport=transport
            )
        except TypeError:
            return httpx.Client(timeout=60, verify=verify, transport=transport)
    return httpx.Client(timeout=60, verify=verify, transport=transport)


_HTTP_CLIENT = build_http_client()
_LLM_CACHE: Dict[Tuple[str, float], ChatOpenAI] = {}


def get_llm(model_name: str, temperature: float = 0.0) -> ChatOpenAI:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set")
    key = (model_name, temperature)
    if key not in _LLM_CACHE:
        _LLM_CACHE[key] = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            api_key=OPENAI_API_KEY,
            http_client=_HTTP_CLIENT,
        )
    return _LLM_CACHE[key]


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_key(text: str) -> str:
    return normalize_whitespace(text).lower()


def sanitize_session_name(name: str) -> str:
    cleaned = normalize_whitespace(name)
    cleaned = cleaned.replace("`", "").replace('"', "'")
    return cleaned


def sanitize_label(value: str, fallback: str) -> str:
    cleaned = _LABEL_SAFE_RE.sub("_", value or "")
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or fallback


def sanitize_properties(properties: Dict[str, Any]) -> Dict[str, Any]:
    cleaned: Dict[str, Any] = {}
    for key, value in (properties or {}).items():
        if isinstance(value, (str, int, float, bool)):
            cleaned[key] = value
        elif isinstance(value, list) and all(
            isinstance(item, (str, int, float, bool)) for item in value
        ):
            cleaned[key] = value
        else:
            cleaned[key] = str(value)
    return cleaned


def strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```", "", cleaned)
    return cleaned.strip()


def normalize_medical_term(term: str) -> str:
    """Normalize medical terminology for better deduplication"""
    term = term.lower().strip()
    # Remove dosage units from names for better deduplication
    # e.g., "ибупрофен 200мг" -> "ибупрофен"
    term = re.sub(r'\s*\d+\s*(мг|г|мл|таб|капс|мкг|ме|ед)\b', '', term)
    # Remove extra whitespace
    term = re.sub(r'\s+', ' ', term).strip()
    return term


def build_entity_id(session_name: str, entity_name: str, entity_type: str) -> str:
    # Normalize medical terms for better deduplication
    normalized_name = normalize_medical_term(entity_name)
    return f"{normalize_key(session_name)}::{normalize_key(entity_type)}::{normalize_key(normalized_name)}"


ENTITY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Ты эксперт по извлечению медицинских сущностей из медицинских документов (фармацевтические инструкции, клинические рекомендации, протоколы лечения).\n\n"
            "ТИПЫ СУЩНОСТЕЙ (для универсальной медицинской системы):\n\n"
            "ПРЕПАРАТЫ И ВЕЩЕСТВА:\n"
            "- Препарат: торговые названия лекарств (Нурофен, Парацетамол, Циклофосфамид)\n"
            "- ДействующееВещество: активные компоненты (ибупрофен, парацетамол)\n"
            "- ФармакологическаяГруппа: классы препаратов (НПВС, антибиотики, цитостатики)\n"
            "- ЛекарственнаяФорма: форма выпуска (таблетки, суспензия, инъекция)\n\n"
            "ЗАБОЛЕВАНИЯ И СОСТОЯНИЯ:\n"
            "- Заболевание: нозологии и диагнозы (солитарная плазмоцитома, гастрит, диабет)\n"
            "- Симптом: клинические проявления (боль, лихорадка, кровотечение, анемия)\n"
            "- Синдром: симптомокомплексы (нефротический синдром, интоксикация)\n"
            "- СтадияЗаболевания: стадии и степени (I стадия, ремиссия, обострение)\n"
            "- КодМКБ: коды МКБ-10 (C90.2, C90.3, E10)\n"
            "- ФизиологическоеСостояние: особые состояния (беременность, лактация, детский возраст)\n\n"
            "ДИАГНОСТИКА:\n"
            "- ДиагностическийМетод: методы обследования (МРТ, КТ, биопсия, УЗИ, рентген)\n"
            "- ЛабораторныйПоказатель: анализы и маркеры (гемоглобин, М-градиент, креатинин)\n"
            "- МедицинскоеОборудование: оборудование (томограф, эндоскоп)\n\n"
            "ЛЕЧЕНИЕ:\n"
            "- МетодЛечения: виды терапии (лучевая терапия, химиотерапия, хирургия)\n"
            "- МедицинскаяПроцедура: процедуры (трепанобиопсия, пункция, резекция)\n"
            "- ПротоколЛечения: схемы лечения (протокол ASCT, режим VRD)\n"
            "- Дозировка: дозы и схемы (200мг 3 раза в день, 40-50 Гр)\n"
            "- Способприменения: как применять (внутрь, в/в, местно, лучевая терапия)\n"
            "- ДлительностьЛечения: сроки (курс 21 день, не более 5 дней)\n\n"
            "ПОКАЗАНИЯ И ПРОТИВОПОКАЗАНИЯ:\n"
            "- Показание: когда применяется (головная боль, локализованная опухоль)\n"
            "- Противопоказание: когда НЕЛЬЗЯ (беременность, почечная недостаточность)\n"
            "- Побочноедействие: нежелательные эффекты (тошнота, миелосупрессия)\n"
            "- Осложнение: тяжелые последствия (кровотечение, инфекция, рецидив)\n\n"
            "ОРГАНИЗАЦИЯ ПОМОЩИ:\n"
            "- МедицинскаяОрганизация: учреждения (онкодиспансер, поликлиника)\n"
            "- Специалист: врачи (онколог, гематолог, хирург)\n"
            "- Возрастнаякатегория: для кого (взрослые, дети 6-12 лет)\n\n"
            "ПРОЧЕЕ:\n"
            "- Производитель: компании-производители\n"
            "- УровеньДоказательности: уровни рекомендаций (A, B, C, 1, 2)\n"
            "- Взаимодействие: взаимодействия (с алкоголем, с антикоагулянтами)\n\n"
            "ПРАВИЛА:\n"
            "1. Извлекай ВСЕ упоминания заболеваний, методов диагностики, лечения\n"
            "2. Для препаратов извлекай дозировки, показания, противопоказания\n"
            "3. Для заболеваний извлекай симптомы, стадии, коды МКБ\n"
            "4. Сохраняй числовые значения и единицы измерения в properties\n"
            "5. Используй точные названия из текста\n\n"
            "Формат ответа - ТОЛЬКО валидный JSON массив:\n"
            "[\n"
            "  {{\"name\": \"Солитарная плазмоцитома\", \"type\": \"Заболевание\", \"properties\": {{\"код_мкб\": \"C90.2\"}}}},\n"
            "  {{\"name\": \"лучевая терапия\", \"type\": \"МетодЛечения\", \"properties\": {{\"доза\": \"40-50 Гр\"}}}},\n"
            "  {{\"name\": \"МРТ\", \"type\": \"ДиагностическийМетод\", \"properties\": {{}}}},\n"
            "  {{\"name\": \"беременность\", \"type\": \"Противопоказание\", \"properties\": {{}}}}\n"
            "]\n\n"
            "Если сущностей нет, верни: []",
        ),
        ("human", "Текст:\n{text}"),
    ]
)


RELATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Ты эксперт по извлечению отношений между медицинскими сущностями (заболевания, препараты, методы диагностики и лечения).\n\n"
            "ТИПЫ ОТНОШЕНИЙ (используй русский язык, глагольные формы):\n\n"
            "ДИАГНОСТИКА:\n"
            "- диагностируется_методом / выявляется_с_помощью\n"
            "- характеризуется_симптомом / проявляется\n"
            "- имеет_стадию / классифицируется_как\n"
            "- кодируется_по_мкб / соответствует_коду\n\n"
            "ЛЕЧЕНИЕ И ПРИМЕНЕНИЕ:\n"
            "- лечится_методом / применяется_терапия\n"
            "- показан_при / применяется_при / назначается_при\n"
            "- эффективен_при / помогает_при / устраняет\n"
            "- включает_процедуру / требует_выполнения\n\n"
            "ПРОТИВОПОКАЗАНИЯ И ОГРАНИЧЕНИЯ:\n"
            "- противопоказан_при / запрещен_при\n"
            "- не_рекомендуется_при / ограничен_при\n"
            "- требует_осторожности_при / с_осторожностью_при\n\n"
            "ПОБОЧНЫЕ ЭФФЕКТЫ И ОСЛОЖНЕНИЯ:\n"
            "- вызывает_побочный_эффект / может_вызвать\n"
            "- приводит_к_осложнению / вызывает_осложнение\n"
            "- осложняется / прогрессирует_в\n\n"
            "ПРЕПАРАТЫ:\n"
            "- содержит_вещество / действующее_вещество\n"
            "- входит_в_группу / относится_к_классу\n"
            "- выпускается_в_форме / имеет_форму\n"
            "- назначается_в_дозе / имеет_дозировку\n"
            "- принимается_способом / применяется\n"
            "- взаимодействует_с / несовместим_с\n\n"
            "КЛИНИЧЕСКИЕ СВЯЗИ:\n"
            "- является_фактором_риска / способствует_развитию\n"
            "- ассоциирован_с / связан_с\n"
            "- дифференцируется_от / отличается_от\n"
            "- трансформируется_в / переходит_в\n\n"
            "ОРГАНИЗАЦИЯ ПОМОЩИ:\n"
            "- выполняется_специалистом / проводится_врачом\n"
            "- оказывается_в_учреждении / доступен_в\n"
            "- рекомендован_для_возраста / предназначен_для\n"
            "- имеет_уровень_доказательности / подтвержден_уровнем\n\n"
            "ПРАВИЛА:\n"
            "1. Создавай отношения ТОЛЬКО между сущностями из списка (используй id)\n"
            "2. Тип отношения = глагольная форма (лечится_методом, а не \"лечение\")\n"
            "3. Добавляй контекст в properties (дозы, сроки, условия)\n"
            "4. Не дублируй отношения\n\n"
            "Формат - валидный JSON:\n"
            "[\n"
            "  {{\"source_id\": \"id1\", \"target_id\": \"id2\", \"type\": \"диагностируется_методом\",\n"
            "   \"properties\": {{\"контекст\": \"для уточнения распространенности\"}}}},\n"
            "  {{\"source_id\": \"id3\", \"target_id\": \"id4\", \"type\": \"лечится_методом\",\n"
            "   \"properties\": {{\"доза\": \"40-50 Гр\"}}}}\n"
            "]\n\n"
            "Если отношений нет: []",
        ),
        (
            "human",
            "Текст:\n{text}\n\n"
            "Сущности (используй только эти id в отношениях):\n{entities}",
        ),
    ]
)


QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Ты отвечаешь на вопрос пользователя на основе извлеченных фрагментов из медицинских документов. "
            "Используй только информацию из предоставленного контекста. "
            "Если данных недостаточно для ответа, честно скажи, что ответа нет в текущих документах. "
            "Отвечай по-русски, структурированно и по делу.",
        ),
        (
            "human",
            "Вопрос:\n{question}\n\n"
            "Контекст из документов:\n{results}",
        ),
    ]
)


def extract_entities_parallel(chunks: List[str], llm: ChatOpenAI) -> List[List[Dict[str, Any]]]:
    chain = ENTITY_PROMPT | llm | StrOutputParser()
    results: List[List[Dict[str, Any]]] = [[] for _ in chunks]

    def _extract(text: str) -> List[Dict[str, Any]]:
        response = chain.invoke({"text": text[:MAX_CHUNK_CHARS]})
        parsed = json_repair.loads(strip_code_fences(response))
        if not isinstance(parsed, list):
            return []
        clean_entities = []
        for entity in parsed:
            if not isinstance(entity, dict):
                continue
            name = str(entity.get("name", "")).strip()
            entity_type = str(entity.get("type", "")).strip()
            if not name or not entity_type:
                continue
            properties = entity.get("properties")
            if not isinstance(properties, dict):
                properties = {}
            properties = sanitize_properties(properties)
            clean_entities.append(
                {"name": name, "type": entity_type, "properties": properties}
            )
        return clean_entities

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_extract, text): idx for idx, text in enumerate(chunks)}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                print(f"Ошибка извлечения сущностей (чанк {idx + 1}): {exc}")
                results[idx] = []

    return results


def extract_relations_parallel(
    chunks: List[str],
    entities_by_chunk: List[List[Dict[str, Any]]],
    llm: ChatOpenAI,
) -> List[Dict[str, Any]]:
    chain = RELATION_PROMPT | llm | StrOutputParser()
    relations: List[Dict[str, Any]] = []

    def _extract(text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if len(entities) < 2:
            return []
        payload = [
            {"id": e["id"], "name": e["name"], "type": e["type"]} for e in entities
        ]
        response = chain.invoke(
            {"text": text[:MAX_CHUNK_CHARS], "entities": json.dumps(payload, ensure_ascii=False)}
        )
        parsed = json_repair.loads(strip_code_fences(response))
        if not isinstance(parsed, list):
            return []
        allowed_ids = {e["id"] for e in entities}
        clean_relations = []
        for rel in parsed:
            if not isinstance(rel, dict):
                continue
            source_id = str(rel.get("source_id", "")).strip()
            target_id = str(rel.get("target_id", "")).strip()
            rel_type = str(rel.get("type", "")).strip()
            if not source_id or not target_id or not rel_type:
                continue
            if source_id == target_id:
                continue
            if source_id not in allowed_ids or target_id not in allowed_ids:
                continue
            properties = rel.get("properties")
            if not isinstance(properties, dict):
                properties = {}
            properties = sanitize_properties(properties)
            clean_relations.append(
                {
                    "source_id": source_id,
                    "target_id": target_id,
                    "type": rel_type,
                    "properties": properties,
                }
            )
        return clean_relations

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}
        for text, entities in zip(chunks, entities_by_chunk):
            futures[executor.submit(_extract, text, entities)] = True
        for future in as_completed(futures):
            try:
                relations.extend(future.result())
            except Exception as exc:
                print(f"Ошибка извлечения отношений: {exc}")

    return relations


def process_document(file_path: str, session_name: str) -> Dict[str, Any]:
    """Process PDF or Markdown document and store in Qdrant"""
    document_name = os.path.splitext(os.path.basename(file_path))[0]

    # Detect file type and use appropriate loader
    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension == '.pdf':
        loader = UnstructuredPDFLoader(file_path)
    elif file_extension in ['.md', '.markdown']:
        loader = UnstructuredMarkdownLoader(file_path)
    else:
        return {
            "status": "error",
            "document": document_name,
            "message": f"Неподдерживаемый формат файла: {file_extension}. Используйте .pdf или .md",
        }

    print(f"[DEBUG] Loading {file_extension} file: {file_path}")
    docs = loader.load()
    print(f"[DEBUG] Loaded {len(docs)} document(s) with total {sum(len(d.page_content) for d in docs)} characters")

    splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)
    chunk_texts = [normalize_whitespace(c.page_content) for c in chunks]
    print(f"[DEBUG] Split into {len(chunks)} chunks")

    ingest_llm = get_llm(OPENAI_INGEST_MODEL, temperature=0.0)

    # Extract entities from chunks
    raw_entities_by_chunk = extract_entities_parallel(chunk_texts, ingest_llm)
    entities_by_chunk: List[List[Dict[str, Any]]] = []

    for chunk_entities in raw_entities_by_chunk:
        prepared_chunk: List[Dict[str, Any]] = []
        for entity in chunk_entities:
            entity_id = build_entity_id(session_name, entity["name"], entity["type"])
            prepared = {
                "id": entity_id,
                "name": entity["name"],
                "type": entity["type"],
                "properties": dict(entity.get("properties", {})),
            }
            prepared_chunk.append(prepared)
        entities_by_chunk.append(prepared_chunk)

    # Store in Qdrant
    qdrant.add_chunks(chunk_texts, session_name, document_name, entities_by_chunk)

    total_entities = sum(len(e) for e in entities_by_chunk)

    print(f"[DEBUG] Stored {len(chunks)} chunks with {total_entities} entities in Qdrant")

    return {
        "status": "success",
        "document": document_name,
        "chunks": len(chunks),
        "entities": total_entities,
        "message": f"Готово: {document_name}",
    }


def get_sessions() -> List[str]:
    return qdrant.get_sessions()


def get_documents(session_name: Optional[str]) -> List[List[Any]]:
    if not session_name:
        return []
    docs = qdrant.get_documents(session_name)
    data = []
    for doc in docs:
        data.append([
            doc["name"],
            "",
            doc["chunks"],
            doc["entities"],
            0
        ])
    return data


def create_session_ui(session_name: str):
    name = sanitize_session_name(session_name)
    if not name:
        return (
            "❌ Укажите название сессии.",
            gr.Dropdown(choices=get_sessions(), allow_custom_value=True),
            [],
        )
    sessions = get_sessions()
    if name not in sessions:
        sessions = [name] + sessions
    return (
        f"✅ Сессия '{name}' готова к использованию.",
        gr.Dropdown(choices=sessions, value=name, allow_custom_value=True),
        get_documents(name),
    )


def on_session_change(session_name: str):
    if not session_name:
        return [], "⚠️ Выберите сессию."
    return get_documents(session_name), f"Текущая сессия: {session_name}"


def upload_pdfs(files, session_name: str, progress=gr.Progress()):
    """Upload and process PDF or Markdown documents"""
    if not session_name:
        return "❌ Сначала выберите сессию.", get_documents(session_name)
    if not files:
        return "❌ Файлы не выбраны.", get_documents(session_name)

    file_list = files if isinstance(files, list) else [files]
    messages = []
    total = len(file_list)

    for idx, file_obj in enumerate(file_list, start=1):
        progress((idx - 1) / total, desc=f"Обработка файла {idx}/{total}...")
        try:
            result = process_document(file_obj.name, session_name)
            if result["status"] == "exists":
                messages.append(f"⚠️ {result['message']}")
            elif result["status"] == "error":
                messages.append(f"❌ {result['message']}")
            else:
                messages.append(
                    f"✅ {result['document']}: "
                    f"{result['chunks']} чанков, "
                    f"{result['entities']} сущностей"
                )
        except Exception as exc:
            messages.append(f"❌ Ошибка при обработке {file_obj.name}: {exc}")

    progress(1.0, desc="Готово!")
    return "\n".join(messages), get_documents(session_name)


def delete_session(session_name: str, confirmed: bool):
    """Delete a session and all its data"""
    if not session_name:
        return (
            "❌ Выберите сессию для удаления.",
            gr.Dropdown(choices=get_sessions(), allow_custom_value=True),
            [],
            False,
        )
    if not confirmed:
        return (
            "⚠️ Подтвердите удаление сессии.",
            gr.Dropdown(choices=get_sessions(), allow_custom_value=True),
            get_documents(session_name),
            False,
        )

    try:
        qdrant.delete_session(session_name)
        remaining_sessions = get_sessions()
        new_session = remaining_sessions[0] if remaining_sessions else None

        return (
            f"✅ Сессия '{session_name}' удалена.",
            gr.Dropdown(
                choices=remaining_sessions,
                value=new_session,
                allow_custom_value=True,
            ),
            get_documents(new_session) if new_session else [],
            False,
        )
    except Exception as e:
        return (
            f"❌ Ошибка удаления: {e}",
            gr.Dropdown(choices=get_sessions(), allow_custom_value=True),
            [],
            False,
        )


def answer_question(question: str, session_name: str) -> Tuple[str, str]:
    """Answer question using Qdrant vector search and return answer + graph HTML"""

    print(f"[DEBUG] Question: {question}")

    # 1. Search Qdrant for similar chunks
    search_results = qdrant.search(question, session_name, k=5)

    print(f"[DEBUG] Found {len(search_results)} relevant chunks")

    if not search_results:
        return "В текущих документах нет данных для ответа.", ""

    # 2. Collect all entities from retrieved chunks
    all_entities = {}
    for chunk_data in search_results:
        if "entities" in chunk_data and chunk_data["entities"]:
            for ent in chunk_data["entities"]:
                ent_key = f"{ent['name']}::{ent['type']}"
                if ent_key not in all_entities:
                    all_entities[ent_key] = ent

    entities = list(all_entities.values())

    # 3. Extract relations from chunks
    llm = get_llm(OPENAI_INGEST_MODEL, temperature=0.0)
    relations = extract_relations_from_chunks(search_results, llm, RELATION_PROMPT)

    print(f"[DEBUG] Extracted {len(entities)} entities, {len(relations)} relations")

    # 4. Build graph HTML
    graph_html = build_graph_html(entities, relations)

    # 5. Generate answer using LLM
    qa_llm = get_llm(OPENAI_QA_MODEL, temperature=0.2)
    chain = QA_PROMPT | qa_llm | StrOutputParser()

    # Format context from chunks
    context = "\n\n".join([
        f"[Документ: {r['document_name']}, Chunk {r['chunk_id']}]\n{r['text']}"
        for r in search_results
    ])

    result_text = f"{context}"
    answer = chain.invoke({"question": question, "results": result_text})

    print(f"[DEBUG] Generated answer")

    return answer, graph_html


def chat(message: str, history: List[Dict[str, str]], session_name: str, current_graph: str = ""):
    if not session_name:
        return history, "", current_graph
    if not message.strip():
        return history, "", current_graph

    history = history or []
    history.append({"role": "user", "content": message})

    answer, graph_html = answer_question(message, session_name)
    history.append({"role": "assistant", "content": answer})

    return history, "", graph_html


with gr.Blocks() as demo:
    with gr.Sidebar():
        gr.Markdown("### Сессии")
        session_name_input = gr.Textbox(
            label="Название новой сессии", placeholder="Например: Фарм_инструкции"
        )
        create_session_btn = gr.Button("Создать сессию")
        session_selector = gr.Dropdown(
            choices=[],
            label="Текущая сессия",
            allow_custom_value=True,
        )
        session_status = gr.Textbox(label="Статус сессии", interactive=False)

        gr.Markdown("### Загрузка документов")
        upload_files = gr.File(file_count="multiple", file_types=[".pdf", ".md", ".markdown"], label="PDF и Markdown файлы")
        upload_btn = gr.Button("Загрузить и обработать")
        upload_status = gr.Textbox(label="Статус загрузки", interactive=False)

        delete_confirm = gr.Checkbox(label="Подтвердить удаление", value=False)
        delete_session_btn = gr.Button("Удалить сессию", variant="stop")
        delete_status = gr.Textbox(label="Статус удаления", interactive=False)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Граф знаний")
            graph_output = gr.HTML(
                label="Knowledge Graph",
                value=(
                    "<div style='padding: 20px; text-align: center; color: #666;'>"
                    "Граф будет отображен после того, как вы зададите вопрос"
                    "</div>"
                ),
            )

        with gr.Column(scale=1):
            gr.Markdown("### Вопросы по документам")
            chatbot = gr.Chatbot(type="messages", height=400)
            user_input = gr.Textbox(placeholder="Спросите о документах...", label="Ваш вопрос")
            ask_btn = gr.Button("Спросить")

            gr.Markdown("### Документы в сессии")
            document_table = gr.Dataframe(
                headers=["Документ", "Создан", "Чанков", "Сущностей", "Связей"],
                datatype=["str", "str", "number", "number", "number"],
                interactive=False,
                value=[],
            )

    create_session_btn.click(
        fn=create_session_ui,
        inputs=[session_name_input],
        outputs=[session_status, session_selector, document_table],
    )
    session_selector.change(
        fn=on_session_change,
        inputs=[session_selector],
        outputs=[document_table, session_status],
    )
    upload_btn.click(
        fn=upload_pdfs,
        inputs=[upload_files, session_selector],
        outputs=[upload_status, document_table],
    )
    delete_session_btn.click(
        fn=delete_session,
        inputs=[session_selector, delete_confirm],
        outputs=[delete_status, session_selector, document_table, delete_confirm],
    )

    ask_btn.click(
        fn=chat,
        inputs=[user_input, chatbot, session_selector, graph_output],
        outputs=[chatbot, user_input, graph_output],
    )
    user_input.submit(
        fn=chat,
        inputs=[user_input, chatbot, session_selector, graph_output],
        outputs=[chatbot, user_input, graph_output],
    )

    def init_ui():
        sessions = get_sessions()
        default_session = sessions[0] if sessions else None
        status = (
            f"Текущая сессия: {default_session}" if default_session else "Создайте сессию."
        )
        return (
            gr.Dropdown(
                choices=sessions,
                value=default_session,
                allow_custom_value=True,
            ),
            get_documents(default_session),
            status,
        )

    demo.load(
        fn=init_ui,
        inputs=None,
        outputs=[session_selector, document_table, session_status],
    )


demo.launch(share=True)
