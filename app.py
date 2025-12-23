import streamlit as st
import vector_store
from langchain_core.messages import HumanMessage, AIMessage
import rag_engine
import os

st.set_page_config(page_title="Internal Research Bot")
st.title("Internal Research Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_db" not in st.session_state:
    st.session_state.vector_db = vector_store.load_vector_db()

with st.sidebar:
    st.write("### Knowledge Base")
    if st.button("Datenbank aktualisieren/neu bauen"):
        with st.spinner("Lese PDFs und erstelle Index..."):
            db, msg = vector_store.create_or_update_vector_db()
            st.session_state.vector_db = db
            st.success(msg)

    if st.session_state.vector_db:
        st.success("Datenbank geladen")
    else:
        st.warning("Keine Datenbank gefunden. Bitte 'Aktualisieren' klicken.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Falls Quellen gespeichert wurden (nur bei assistant messages)
        if "sources" in message:
            with st.expander("Quellen anzeigen"):
                for source in message["sources"]:
                    st.markdown(f"- 📄 **{source['file']}** (Seite {source['page']})")

if prompt := st.chat_input("Was möchtest du wissen?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analysiere Reports..."):
            chat_history_for_chain = []
            for msg in st.session_state.messages[-6:]:
                if msg["role"] == "user":
                    chat_history_for_chain.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    chat_history_for_chain.append(AIMessage(content=msg["content"]))

            response_payload = rag_engine.get_llm_response(
                prompt,
                st.session_state.vector_db,
                chat_history_for_chain
            )

            answer_text = response_payload["answer"]
            source_docs = response_payload["context"]

            st.markdown(answer_text)

            unique_sources = []
            seen_sources = set()

            for doc in source_docs:
                file_name = os.path.basename(doc.metadata.get("source", "Unbekannt"))
                page_num = doc.metadata.get("page", 0) + 1

                source_id = f"{file_name}-{page_num}"

                if source_id not in seen_sources:
                    unique_sources.append({"file": file_name, "page": page_num})
                    seen_sources.add(source_id)

            if unique_sources:
                with st.expander("Quellen anzeigen"):
                    for source in unique_sources:
                        st.markdown(f"- 📄 **{source['file']}** (Seite {source['page']})")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer_text,
        "sources": unique_sources
    })