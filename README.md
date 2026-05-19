# Document Q&A Tool

A RAG-powered document Q&A application that lets you ask natural language 
questions about any PDF document and get answers grounded in the document content.

## Built with
- LangChain — document loading, chunking, and retrieval pipeline
- Anthropic Claude API — answer generation
- Chroma — local vector database for storing embeddings
- HuggingFace Embeddings — converting text chunks to vectors
- Streamlit — web interface

## How it works
1. A PDF is loaded and split into 500-token chunks
2. Each chunk is converted to a vector using a local embedding model
3. Chunks are stored in a Chroma vector database
4. When a question is asked, the most relevant chunks are retrieved
5. Claude uses those chunks to generate a grounded answer

## How to run
1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Add your Anthropic API key as an environment variable
4. Run ingestion: `python ingest.py`
5. Launch the app: `python -m streamlit run app.py`
