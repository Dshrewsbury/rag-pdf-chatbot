import uuid
import time

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_cpp import Llama
from chunk_strategies.utils import chunk

# Dont think i need separate batches, as i am not doing much processing outside of the llm
# This might have been to avoid memory overload or ensure that only a certain amount of data is passed through at a time.
# Essentially might have been memory constraints?

# Initialize the language model (Llama) for embedding generation, specifying the model path,
# enabling embedding mode, setting verbosity, and defining the batch size for processing.
# n_batch = how many documents/chunks the embedding model can process at once
llm = Llama(
    model_path="../models/mxbai-embed-large-v1-f16.gguf",
    embedding=True,
    verbose=False,
    n_batch=512
)

# Define the path to the PDF file and load the document's data.
file = "../data/llama2.pdf"
loader = PyPDFLoader(file)
data = loader.load()

# Set up a text splitter with recursive chunking, where:
# - chunk_size defines the maximum character count for each chunk.
# - chunk_overlap defines the overlap between chunks to preserve context continuity.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512, chunk_overlap=50)
documents = text_splitter.split_documents(data)
print(len(documents))  # Output the total number of generated chunks for reference.

# Define batch parameters for processing chunks in manageable sets.
# batch_size = number of documents processed at once for my code
batch_size = 500
documents_embeddings = []
batches = list(chunk(documents, batch_size))  # Divide documents into batches of the defined size.

# Time the embedding generation process.
start = time.time()
for batch in batches:
    # Generate embeddings for each chunk in the batch by passing text content from each document chunk.
    embeddings = llm.create_embedding([item.page_content for item in batch])

    # Store each document chunk and its corresponding embedding in a list.
    documents_embeddings.extend(
        [
            (document, embeddings['embedding'])
            for document, embeddings in zip(batch, embeddings['data'])
        ]
    )
end = time.time()

# Calculate and output the speed of embedding generation in terms of characters processed per second.
char_per_second = len(''.join([item.page_content for item in documents])) / (end - start)
print(f"Time taken: {end - start:.2f} seconds / {char_per_second:,.2f} characters per second")

# Initialize the Qdrant vector database client, specifying the storage path for embeddings.
client = QdrantClient(path="../embeddings/recursive")

# Create a collection in the vector database with parameters:
# - collection_name specifies the collection identifier.
# - vectors_config defines the vector size and distance metric (Cosine similarity in this case).
client.create_collection(
    collection_name="recursive",
    vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
)

# Store document chunks and their embeddings in the Qdrant vector database.
# Each chunk is stored as a unique point with:
# - a unique identifier.
# - the embedding vector.
# - a payload containing the chunk's text content.
points = [
    PointStruct(
        id=str(uuid.uuid4()),
        vector=embeddings,
        payload={
            "text": doc.page_content
        }
    )
    for doc, embeddings in documents_embeddings
]

# Insert all document chunks with embeddings into the specified Qdrant collection,
# and wait for the operation to complete to confirm successful storage.
operation_info = client.upsert(
    collection_name="recursive",
    wait=True,
    points=points
)
