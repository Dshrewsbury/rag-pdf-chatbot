# rag-pdf-chatbot


## About the Project

A retrieval-augmented generation (RAG) system designed to read, understand, and answer questions about a given research paper in PDF format.

This RAG chatbot:

- Reads and processes PDFs: Extracts and answers questions about content from research papers.
- Memory management: Uses SQLite for both short-term memory and long-term memory.
- Qdrant embeddings: Utilizes Qdrant as a vector database for efficient retrieval
- Flexible chunking: Implements both recursive and semantic chunking
- RESTful API

### Prerequisites
- Docker installed ([Docker Install Guide](https://docs.docker.com/get-docker/))
- Docker Compose installed ([Docker Compose Install Guide](https://docs.docker.com/compose/install/))
- LLM Model downloaded into models folder (https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF)
- Embeddings Model downloaded into models folder (https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1)


## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/Dshrewsbury/rag-pdf-chatbot.git
   cd rag-pdf-chatbot
   docker-compose up --build

 Visit the app at:  http://localhost:8000

## Upcoming Work

Planned improvements include:

- Evaluation Metrics: Adding methods to assess response accuracy and relevance.
- Hierarchical Document Processing: Enhancing chunking to better capture the PDF’s internal structure.
- Adaptive Retrieval and Agentic Chunking: Dynamically adjusting retrieval and chunking based on user interactions.
- Entity Extraction and Summarization: Capturing key entities for memory and providing concise summaries.
- Multi-modal Retrieval and Hybrid Search: Supporting diverse document types and combining search methods for improved accuracy.
- Re-ranking: Prioritizing the most relevant chunks for each query.
