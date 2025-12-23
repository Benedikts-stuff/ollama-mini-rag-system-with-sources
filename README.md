# ollama-mini-rag-system-with-sources
A locally running small language model (from Ollama) combined with a vector Database (Chroma) for efficient, secure browsing through files. 

## Usage
1. Download Ollama and run it. Download the model you want to use. Use command "ollama list" to check what model is currently running. Your model should appear.
2. Change the model name that you want to use in rag_engine.py. My default is:
   llm = ChatOllama(
        model="gemma3:4b",
        temperature=0.1,
        num_ctx=16384,
        keep_alive= "5m"
    )
4. Install requirements: "pip install -r requirements.txt"
5. Fill the Folder "Files" with PDFs (or different text) that you want to use as a knowledge base for the Ollama model.
6. Run the app with "streamlit run app.py". On first start the UI will prompt you to initialize the Database with a button on the sidebar. This Database is used as knowledge base for the RAG systsem. 
7. Have fun!
   
