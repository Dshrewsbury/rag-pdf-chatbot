import sys
import time
from uuid import UUID
from typing import Any, Dict, List
from langchain.callbacks.base import BaseCallbackHandler
from llama_cpp import Llama
from qdrant_client import QdrantClient
from src.memory_database import MemoryDatabaseManager

embedding_llm = Llama(
    model_path="./models/mxbai-embed-large-v1-f16.gguf",
    embedding=True,
    verbose=False,
    n_batch=512,
    max_tokens=512
)
memory_db = MemoryDatabaseManager(db_path="chat_history.db", embedding_llm=embedding_llm)
client = QdrantClient(path="./embeddings/recursive")

class StreamingWebCallbackHandler(BaseCallbackHandler):
    tokens: List[str] = []
    is_responding: bool = False
    response_id: str
    response: str = None
    start_time: float = None


    def on_llm_new_token(self, token: str, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any) -> Any:
        sys.stdout.write(token)
        sys.stdout.flush()
        self.tokens.append(token)

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], *, run_id: UUID, parent_run_id: UUID | None = None, tags: List[str] | None = None, metadata: Dict[str, Any] | None = None, **kwargs: Any) -> Any:
        self.is_responding = True
        self.response_id = run_id
        self.response = None
        self.start_time = time.time()
        self.question = inputs.get("input", "")
        self.original_question = self.question

        # Long-term memory management. Could easily alter to remove short-term memory
        # and just use this as a lookup to avoid global variable usage
        # If time allows (probably wont), add entity extraction
        # user_input_embedding = memory_db.get_local_embedding(question)
        # top_similar = memory_db.find_top_n_similar(user_input_embedding, n=1)
        # if not top_similar.empty:
        #     similar_results = "\n".join(top_similar['user_input_plus_response'].values)
        #     # print(f"Similar past interactions found:\n{similar_results}\n")
        #     # Optionally, append this to the current input for more context
        #     question += f"\nLONG MEMORY: {similar_results}"

        # RAG Retrieval - Search for similar vectors to query
        # Could add summarization due to context limitations
        query_vector = embedding_llm.create_embedding(self.question)['data'][0]['embedding']
        search_results = client.search(
            collection_name="recursive",
            query_vector=query_vector,
            limit=1
        )

        context = "\n\n".join([row.payload['text'] for row in search_results])
        prompt = f"\n\nContext: {context}" + "User: " + self.question
        inputs["input"] = prompt

    def on_chain_end(self, outputs: Dict[str, Any], *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any) -> Any:
        self.is_responding = False
        self.response = outputs['response']
        end_time = time.time()
        latency = end_time - self.start_time

        # Log the question, response, and latency
        self.log_interaction(self.original_question, self.response, latency)

        # Save the interaction and embeddings into the database after generating the response
        # Obviously sucks to parse out context here, needs refactor
        user_input_plus_response = f"{self.question}\n\n{self.response}"
        embeddings = memory_db.get_local_embedding(user_input_plus_response)
        memory_db.save_to_database(user_input_plus_response, embeddings)

        self.question = ""
        self.original_question = ""


    def get_response(self) -> str:
        response_result = self.response
        self.response = None
        
        return response_result

    def log_interaction(self, question: str, response: str, latency: float):
        """Log the interaction details to a text file."""
        with open("chat_log.txt", "a") as f:
            f.write(f"Question: {question}\n")
            f.write(f"Response: {response}\n")
            f.write(f"Latency: {latency:.2f} seconds\n\n")