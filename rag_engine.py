
# rag_engine.py

import os
import json
import hashlib
from pathlib import Path

import chromadb
import networkx as nx
from pypdf import PdfReader
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

load_dotenv()

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / "docs" / "spotify_web_app_architecture.pdf"
DB_DIR = BASE_DIR / "chroma_db"
GRAPH_PATH = DB_DIR / "knowledge_graph.json"

DB_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

# --------------------------------------------------
# 2. Initialize databases and models
# --------------------------------------------------

chroma_client = chromadb.PersistentClient(path=str(DB_DIR))

collection = chroma_client.get_or_create_collection(
    name="spotify_knowledge",
    metadata={"hnsw:space": "cosine"},
)

embedder = SentenceTransformer("all-MiniLM-L6-v2")

api_key = os.getenv("OPENAI_API_KEY")
llm_client = OpenAI(api_key=api_key) if api_key else None

# Each edge records the source chunk that supports it.
KG = nx.MultiDiGraph()


def load_graph():
    """Restore graph relationships after an app restart."""
    if not GRAPH_PATH.exists():
        return

    try:
        data = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))

        for edge in data:
            KG.add_edge(
                edge["subject"],
                edge["object"],
                relation=edge["relation"],
                chunk_id=edge["chunk_id"],
                source_hash=edge["source_hash"],
                source=edge["source"],
                page=edge["page"],
            )
    except (OSError, ValueError, KeyError):
        KG.clear()


def save_graph():
    """Persist graph edges to JSON."""
    edges = []

    for subject, obj, _, data in KG.edges(
        keys=True, data=True
    ):
        edges.append({
            "subject": subject,
            "object": obj,
            **data,
        })

    GRAPH_PATH.write_text(
        json.dumps(edges, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


load_graph()


# --------------------------------------------------
# 3. PDF processing using PyPDF
# --------------------------------------------------

def load_pdf_chunks(
    pdf_path=PDF_PATH,
    chunk_size=180,
    overlap=30,
):
    """Extract PDF text page by page and create chunks."""

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}\n"
            "Ensure the file is inside the docs folder."
        )

    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "Use chunk_size > overlap >= 0."
        )

    reader = PdfReader(str(pdf_path))
    chunks = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        words = text.split()

        step = chunk_size - overlap

        for start in range(0, len(words), step):
            part = words[start:start + chunk_size]

            if not part:
                continue

            chunks.append({
                "text": " ".join(part),
                "page": page_number,
                "source": pdf_path.name,
            })

    if not chunks:
        raise ValueError(
            "No extractable text found. "
            "If the PDF contains scanned pages, OCR is required."
        )

    return chunks


# --------------------------------------------------
# 4. Knowledge graph extraction
# --------------------------------------------------

def extract_relationships(text):
    """Extract explicit subject-relation-object triples."""

    if llm_client is None:
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Add it to your .env file."
        )

    prompt = f"""
Extract factual relationships explicitly supported by this text.

Return valid JSON in this format:
{{
  "relations": [
    {{
      "subject": "Library & Playlist Service",
      "relation": "stores data in",
      "object": "PostgreSQL"
    }}
  ]
}}

Rules:
- Do not invent facts or relationships.
- Use concise, consistent entity names.
- Keep the relation short.
- Return at most 12 relationships.
- If no relationships are supported, return an empty list.

TEXT:
{text}
"""

    response = llm_client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
    )

    result = json.loads(
        response.choices[0].message.content or "{}"
    )

    relations = result.get("relations", [])

    if not isinstance(relations, list):
        return []

    return relations


# --------------------------------------------------
# 5. Index the PDF into ChromaDB and the graph
# --------------------------------------------------

def ingest_knowledge_pdf(pdf_path=PDF_PATH):
    """Index the PDF's chunks and extract graph relationships."""

    pdf_path = Path(pdf_path)
    chunks = load_pdf_chunks(pdf_path)

    source_hash = hashlib.sha256(
        pdf_path.read_bytes()
    ).hexdigest()

    # Remove previous vector records for this exact document.
    old = collection.get(
        where={"source_hash": source_hash},
        include=["metadatas"],
    )

    if old["ids"]:
        collection.delete(ids=old["ids"])

    # Remove graph edges previously created for this document.
    edges_to_remove = [
        (s, o, key)
        for s, o, key, data in KG.edges(
            keys=True, data=True
        )
        if data.get("source_hash") == source_hash
    ]

    KG.remove_edges_from(edges_to_remove)

    ids = []
    documents = []
    metadatas = []

    for index, chunk in enumerate(chunks):
        chunk_id = f"{source_hash[:16]}_{index}"

        ids.append(chunk_id)
        documents.append(chunk["text"])
        metadatas.append({
            "source": chunk["source"],
            "source_hash": source_hash,
            "page": chunk["page"],
            "chunk_index": index,
        })

    embeddings = embedder.encode(
        documents,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).tolist()

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    # Extract and store graph relationships.
    for chunk_id, chunk in zip(ids, chunks):
        relationships = extract_relationships(chunk["text"])

        for rel in relationships:
            if not isinstance(rel, dict):
                continue

            subject = str(rel.get("subject", "")).strip()
            relation = str(rel.get("relation", "")).strip()
            obj = str(rel.get("object", "")).strip()

            if not subject or not relation or not obj:
                continue

            KG.add_edge(
                subject,
                obj,
                relation=relation,
                chunk_id=chunk_id,
                source_hash=source_hash,
                source=chunk["source"],
                page=chunk["page"],
            )

    save_graph()

    return {
        "chunks": len(chunks),
        "nodes": KG.number_of_nodes(),
        "edges": KG.number_of_edges(),
        "source": pdf_path.name,
    }


# --------------------------------------------------
# 6. Hybrid retrieval: vector + knowledge graph
# --------------------------------------------------

def retrieve(question, k=5):
    """Retrieve relevant passages from vectors and graph edges."""

    if collection.count() == 0:
        return []

    query_embedding = embedder.encode(
        [question],
        normalize_embeddings=True,
    ).tolist()

    vector_results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    evidence = {}

    # Vector results are inserted first.
    docs = vector_results.get("documents") or [[]]
    metas = vector_results.get("metadatas") or [[]]
    distances = vector_results.get("distances") or [[]]

    for doc, meta, distance in zip(
        docs[0], metas[0], distances[0]
    ):
        evidence_key = (
            meta["source_hash"],
            meta["chunk_index"],
        )

        evidence[evidence_key] = {
            "text": doc,
            "source": meta["source"],
            "page": meta["page"],
            "score": round(1 - distance, 4),
            "method": "Vector",
        }

    # Find graph entities mentioned in the question.
    question_lower = question.lower()

    matched_nodes = [
        node for node in KG.nodes
        if str(node).lower() in question_lower
    ]

    # Retrieve evidence from both incoming and outgoing edges.
    graph_chunk_ids = set()

    for node in matched_nodes:
        for _, _, data in KG.out_edges(node, data=True):
            graph_chunk_ids.add(data["chunk_id"])

        for _, _, data in KG.in_edges(node, data=True):
            graph_chunk_ids.add(data["chunk_id"])

    if graph_chunk_ids:
        graph_results = collection.get(
            ids=list(graph_chunk_ids),
            include=["documents", "metadatas"],
        )

        for doc, meta in zip(
            graph_results["documents"],
            graph_results["metadatas"],
        ):
            evidence_key = (
                meta["source_hash"],
                meta["chunk_index"],
            )

            # Preserve vector ranking when a chunk occurs in both.
            if evidence_key not in evidence:
                evidence[evidence_key] = {
                    "text": doc,
                    "source": meta["source"],
                    "page": meta["page"],
                    "score": None,
                    "method": "Knowledge Graph",
                }
            else:
                evidence[evidence_key]["method"] = "Vector + Graph"

    # Return vector evidence first, followed by graph-only evidence.
    results = list(evidence.values())
    results.sort(
        key=lambda item: (
            item["score"] is None,
            -(item["score"] or 0),
        )
    )

    return results[:k + 3]


# --------------------------------------------------
# 7. Generate a grounded answer
# --------------------------------------------------

def answer_question(question):
    """Generate an answer and return its supporting passages."""

    if llm_client is None:
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Configure your .env file."
        )

    sources = retrieve(question)

    if not sources:
        return (
            "No relevant evidence was found in the knowledge base. "
            "Process the PDF first or try a more specific question.",
            [],
        )

    context_parts = []

    for index, source in enumerate(sources, start=1):
        context_parts.append(
            f"[Source {index}: {source['source']}, "
            f"page {source['page']}]\n{source['text']}"
        )

    context = "\n\n".join(context_parts)

    response = llm_client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer using only the supplied evidence. "
                    "Cite factual claims using [Source 1], "
                    "[Source 2], etc. Use only source numbers "
                    "that exist in the evidence. If the evidence "
                    "does not answer the question, say so. "
                    "Do not invent facts."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"QUESTION:\n{question}\n\n"
                    f"EVIDENCE:\n{context}"
                ),
            },
        ],
    )

    answer = response.choices[0].message.content or (
        "No answer was generated."
    )

    return answer, sources
