
# app.py

import streamlit as st

from rag_engine import (
    PDF_PATH,
    KG,
    collection,
    ingest_knowledge_pdf,
    answer_question,
)

# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Hybrid KG-RAG Demo",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Hybrid KG-RAG Knowledge Assistant")

st.markdown(
    """
    Ask questions about your Spotify-like web application
    architecture using vector retrieval and a knowledge graph.
    """
)

# --------------------------------------------------
# 2. Knowledge base status
# --------------------------------------------------

with st.sidebar:
    st.header("Knowledge Base")

    st.caption("PDF document")
    st.code(str(PDF_PATH), language=None)

    if PDF_PATH.exists():
        st.success("Knowledge PDF found")
    else:
        st.error("Knowledge PDF not found")

    st.divider()

    if st.button(
        "Process Knowledge PDF",
        type="primary",
        use_container_width=True,
        disabled=not PDF_PATH.exists(),
    ):
        try:
            with st.spinner(
                "Extracting PDF, creating embeddings, "
                "and building the knowledge graph..."
            ):
                result = ingest_knowledge_pdf()

            st.session_state["ingestion_result"] = result
            st.success("Knowledge base processed successfully!")

        except Exception as exc:
            st.error(f"Processing failed: {exc}")

    st.divider()

    st.subheader("Database Statistics")

    st.metric(
        "Vector chunks",
        collection.count(),
    )

    st.metric(
        "Graph entities",
        KG.number_of_nodes(),
    )

    st.metric(
        "Graph relationships",
        KG.number_of_edges(),
    )

    if "ingestion_result" in st.session_state:
        st.caption(
            f"Last processed: "
            f"{st.session_state['ingestion_result']['source']}"
        )

# --------------------------------------------------
# 3. Main question-answer interface
# --------------------------------------------------

st.subheader("Ask your knowledge base")

question = st.text_input(
    "Enter your question",
    placeholder=(
        "Which database stores playlist information, "
        "and which service manages playlists?"
    ),
)

ask_button = st.button(
    "Ask Question",
    type="primary",
    disabled=not question.strip(),
)

if ask_button:
    try:
        with st.spinner(
            "Retrieving relevant passages and generating an answer..."
        ):
            answer, sources = answer_question(question.strip())

        st.subheader("Answer")
        st.markdown(answer)

        st.divider()
        st.subheader("Retrieved Sources")

        if sources:
            for index, source in enumerate(sources, start=1):
                method = source["method"]
                page = source["page"]
                filename = source["source"]

                with st.expander(
                    f"Source {index} | Page {page} | "
                    f"{method} | {filename}"
                ):
                    st.markdown("**Extracted passage**")
                    st.write(source["text"])

                    if source["score"] is not None:
                        st.caption(
                            "Vector similarity score: "
                            f"{source['score']:.4f}"
                        )
                    else:
                        st.caption(
                            "Retrieved through graph relationships"
                        )
        else:
            st.info("No supporting passages were retrieved.")

    except Exception as exc:
        st.error(f"Question answering failed: {exc}")

# --------------------------------------------------
# 4. Knowledge graph inspection
# --------------------------------------------------

with st.expander("Inspect knowledge graph relationships"):
    if KG.number_of_edges() == 0:
        st.info(
            "No graph relationships available. "
            "Process the PDF to extract relationships."
        )
    else:
        graph_rows = []

        for subject, obj, data in KG.edges(data=True):
            graph_rows.append({
                "Subject": subject,
                "Relationship": data.get("relation", ""),
                "Object": obj,
                "Source": data.get("source", ""),
                "Page": data.get("page", ""),
            })

        st.dataframe(
            graph_rows,
            use_container_width=True,
            hide_index=True,
        )

st.divider()

st.caption(
    "Demo pipeline: PyPDF → Chunking → Sentence Transformers "
    "→ ChromaDB + NetworkX → Hybrid Retrieval → OpenAI LLM"
)
