import uuid

import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_cpp import Llama
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sklearn.metrics.pairwise import cosine_similarity

"""
Note: Currently not working due to returning far too much, leading to a context window that is too big

Essentially since its putting together semantically similar chunks it winds up
with chunks that are too big for the context

Also overlap and recursive chunking is unnecessary as the goal is to group together semantically similar chunks
using uuid as a unique identifier for each chunk
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

# Split PDF content into initial, fixed-size chunks using RecursiveCharacterTextSplitter.
# The parameters here create an initial "fixed-size" chunking:
# - chunk_size: defines each chunk's maximum size of 1500 characters.
# - chunk_overlap: introduces a 100-character overlap to improve continuity between chunks.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,  # Target chunk size
    chunk_overlap=100  # Overlap between chunks
)
documents = text_splitter.split_documents(data)
print(f"{len(documents)} chunks were created")

# Store the content of initial chunks as text
chunks = [doc.page_content for doc in documents]

# Generate embeddings for each initial fixed-size chunk using the Llama model.
chunk_embeddings = llm.create_embedding(chunks)['data']


# Calculate cosine distances between adjacent chunk embeddings to detect semantic boundaries.
# High cosine distance between two embeddings indicates a likely change in topic.
def calculate_cosine_distances(chunk_embeddings):
    distances = []
    for i in range(len(chunk_embeddings) - 1):
        embedding_current = chunk_embeddings[i]['embedding']
        embedding_next = chunk_embeddings[i + 1]['embedding']
        similarity = cosine_similarity([embedding_current], [embedding_next])[0][0]
        distance = 1 - similarity
        distances.append(distance)
    return distances


# Calculate distances between embeddings of each adjacent chunk
distances = calculate_cosine_distances(chunk_embeddings)

# Set a distance threshold to identify significant semantic breaks.
# The threshold is set to the 95th percentile to flag the most substantial topic changes.
breakpoint_percentile_threshold = 95
breakpoint_distance_threshold = np.percentile(distances, breakpoint_percentile_threshold)
indices_above_thresh = [i for i, x in enumerate(distances) if x > breakpoint_distance_threshold]

# Group initial chunks into larger "semantic chunks" based on topic similarity.
# This step creates variable-length chunks where each chunk is a coherent, larger topic-based grouping.
# These new "semantic chunks" may exceed 1500 characters because they combine multiple smaller chunks until
# a semantic break (distance threshold) is encountered.
start_index = 0
semantic_chunks = []
for index in indices_above_thresh:
    end_index = index
    group = chunks[start_index:end_index + 1]  # Group chunks until a breakpoint
    combined_text = ' '.join(group)  # Concatenate text of grouped chunks
    semantic_chunks.append(combined_text)  # Store the new, larger semantic chunk
    start_index = index + 1  # Move to the next section after the breakpoint

# Append any remaining text as a final semantic chunk
if start_index < len(chunks):
    combined_text = ' '.join(chunks[start_index:])
    semantic_chunks.append(combined_text)

# By this point, the semantic_chunks list contains chunks of varying sizes, many of which may be larger than 1500 characters.

# Generate embeddings for the new, larger semantic chunks using the Llama model
semantic_chunk_embeddings = llm.create_embedding(semantic_chunks)['data']

# Initialize Qdrant client and create a collection to store these semantic chunks.
client = QdrantClient(path="../embeddings/semantic")

client.create_collection(
    collection_name="semantic",
    vectors_config=VectorParams(size=len(semantic_chunk_embeddings[0]['embedding']), distance=Distance.COSINE),
)

# Store each semantic chunk and its corresponding embedding into the Qdrant collection.
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

# Insert all semantic chunks into the Qdrant vector database.
operation_info = client.upsert(
    collection_name="semantic",
    wait=True,
    points=points
)

print(f"Stored {len(points)} semantic chunks in the vector database.")
