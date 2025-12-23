import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever


load_dotenv()

def get_llm_response(query, vector_db, chat_history):
    if not vector_db:
        return "Bitte zuerst die Datenbank initialisieren (Button in der Sidebar)."

    llm = ChatOllama(
        model="gemma3:4b",
        temperature=0.1,
        num_ctx=16384,
        keep_alive= "5m"
    )

    retriever = vector_db.as_retriever(search_kwargs={"k": 5})

    contextualize_q_system_prompt = (
        "Gegeben ein Chatverlauf und die neueste Nutzerfrage, "
        "die sich auf den Verlauf beziehen könnte: "
        "Formuliere eine eigenständige Frage, die auch ohne den Verlauf verständlich ist. "
        "Beantworte die Frage NICHT, sondern formuliere sie nur um oder gib sie unverändert zurück."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    qa_system_prompt = (
        "Du bist ein Research Assistant. "
        "Beantworte die Frage NUR basierend auf dem folgenden Kontext. "
        "Fasse die Informationen präzise zusammen. "
        "Antworte auf Deutsch."
        "\n\n"
        "Kontext:\n{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    response = rag_chain.invoke({
        "input": query,
        "chat_history": chat_history
    })

    return response