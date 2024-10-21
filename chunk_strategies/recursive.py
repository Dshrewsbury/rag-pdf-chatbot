import uuid
import time

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_cpp import Llama
from chunk_strategies.utils import chunk

llm = Llama(
    model_path="../models/mxbai-embed-large-v1-f16.gguf",
    embedding=True,
    verbose=False,
    n_batch=512
)

file = "../data/llama2.pdf"
loader = PyPDFLoader(file)
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500, chunk_overlap=100)
documents = text_splitter.split_documents(data)
print(len(documents))

batch_size = 500
documents_embeddings = []
batches = list(chunk(documents, batch_size))
start = time.time()
for batch in batches:
  embeddings = llm.create_embedding([item.page_content for item in batch])
  documents_embeddings.extend(
    [
      (document, embeddings['embedding'])
      for document, embeddings in zip(batch, embeddings['data'])
    ]
  )
end = time.time()
char_per_second = len(''.join([item.page_content for item in documents])) / (end-start)
print(f"Time taken: {end-start:.2f} seconds / {char_per_second:,.2f} characters per second")

# Init client and create collection
client = QdrantClient(path="../embeddings/recursive")

client.create_collection(
    collection_name="recursive",
    vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
)

# Store documemts
points = [
  PointStruct(
    id = str(uuid.uuid4()),
    vector = embeddings,
    payload = {
      "text": doc.page_content
    }
  )
  for doc, embeddings in documents_embeddings
]

operation_info = client.upsert(
    collection_name="recursive",
    wait=True,
    points=points
)
