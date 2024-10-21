import uuid
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_cpp import Llama

"""
Note: Currently not working due to returning far too much, leading to a context window that is too big
"""


# Load PDF
file = "../data/llama2.pdf"
loader = PyPDFLoader(file)
data = loader.load()

# Load Llama embedding model
llm = Llama(
    model_path="../models/mxbai-embed-large-v1-f16.gguf",
    embedding=True,
    verbose=False,
    n_batch=512
)

# Split PDF content into fixed-size chunks using RecursiveCharacterTextSplitter
# Could probably just do more naive splitting for semantic chunking rather than making this semantic a bit of a hybrid
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,  # The size of each chunk in characters
    chunk_overlap=100  # The overlap between consecutive chunks
)
documents = text_splitter.split_documents(data)
print(f"{len(documents)} chunks were created")

# Store chunks as text
chunks = [doc.page_content for doc in documents]

# Generate embeddings for each chunk using Llama model
chunk_embeddings = llm.create_embedding(chunks)['data']

# Calculate cosine distances between adjacent chunk embeddings
def calculate_cosine_distances(chunk_embeddings):
    distances = []
    for i in range(len(chunk_embeddings) - 1):
        embedding_current = chunk_embeddings[i]['embedding']
        embedding_next = chunk_embeddings[i + 1]['embedding']
        similarity = cosine_similarity([embedding_current], [embedding_next])[0][0]
        distance = 1 - similarity
        distances.append(distance)
    return distances

distances = calculate_cosine_distances(chunk_embeddings)

# Determine breakpoints based on the distance threshold
breakpoint_percentile_threshold = 95
breakpoint_distance_threshold = np.percentile(distances, breakpoint_percentile_threshold)
indices_above_thresh = [i for i, x in enumerate(distances) if x > breakpoint_distance_threshold]

# Split chunks further based on semantic similarity
start_index = 0
semantic_chunks = []
for index in indices_above_thresh:
    end_index = index
    group = chunks[start_index:end_index + 1]
    combined_text = ' '.join(group)
    semantic_chunks.append(combined_text)
    start_index = index + 1
if start_index < len(chunks):
    combined_text = ' '.join(chunks[start_index:])
    semantic_chunks.append(combined_text)


# Generate embeddings for the new semantic chunks using Llama model
semantic_chunk_embeddings = llm.create_embedding(semantic_chunks)['data']

# Initialize Qdrant client and create collection
# Put different types of chunking under same collection?
client = QdrantClient(path="../embeddings/semantic")

client.create_collection(
    collection_name="semantic",
    vectors_config=VectorParams(size=len(semantic_chunk_embeddings[0]['embedding']), distance=Distance.COSINE),
)

# Store semantic chunks and their embeddings into Qdrant
points = [
    PointStruct(
        id=str(uuid.uuid4()),
        vector=embedding['embedding'],
        payload={
            "text": chunk
        }
    )
    for chunk, embedding in zip(semantic_chunks, semantic_chunk_embeddings)
]

operation_info = client.upsert(
    collection_name="semantic",
    wait=True,
    points=points
)

print(f"Stored {len(points)} semantic chunks in the vector database.")
