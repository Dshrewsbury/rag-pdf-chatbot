# rag-pdf-chatbot


## About the Project

A chatbot that can accurately and truthfully(well, relatively) answer
queries related to the given pdf.

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

 Note: Building the image takes some time. Also, the user field on the app is currently required to enter a prompt.
