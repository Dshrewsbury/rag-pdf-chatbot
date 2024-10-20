import sys
import time
from uuid import UUID
from typing import Any, Dict, List
from langchain.callbacks.base import BaseCallbackHandler
from llama_cpp import Llama
from qdrant_client import QdrantClient
from src.memory_database import MemoryDatabaseManager

class StreamingWebCallbackHandler(BaseCallbackHandler):
    tokens: List[str] = []
    is_responding: bool = False
    response_id: str
    response: str = None
    start_time: float = None
    # embedding_llm = Llama(
    #     model_path="./models/mxbai-embed-large-v1-f16.gguf",
    #     embedding=True,
    #     verbose=False,
    #     n_batch=512,
    #     max_tokens=512
    # )
    # memory_db = MemoryDatabaseManager(db_path="chat_history.db", embedding_llm=embedding_llm)
    # client = QdrantClient(path="test_embeddings")

    def on_llm_new_token(self, token: str, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any) -> Any:
        """Run on new LLM token. Only available when streaming is enabled."""
        sys.stdout.write(token)
        sys.stdout.flush()
        self.tokens.append(token)

    # Bit weird to do embedding logic in on_chain_start
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], *, run_id: UUID, parent_run_id: UUID | None = None, tags: List[str] | None = None, metadata: Dict[str, Any] | None = None, **kwargs: Any) -> Any:
        self.is_responding = True
        self.response_id = run_id
        self.response = None
        self.start_time = time.time()  # Capture the start time of the chain
        self.question = inputs.get("input", "")  # Capture the user input
        self.original_question = self.question
        # print("CHAIN: ", self.original_question)
        #
        # query_vector = self.embedding_llm.create_embedding(self.original_question)['data'][0]['embedding']
        # search_results = self.client.search(
        #     collection_name="podcast",
        #     query_vector=query_vector,
        #     limit=5
        # )
        # print(f"Context:\n{search_results}\n")
        # context = "\n\n".join([row.payload['text'] for row in search_results])
        #question += f"\nCONTEXT: {context}"
       #self.question += f"\nCONTEXT: Brand Engagement Network is an AI company that many people want to work for."


    def on_chain_end(self, outputs: Dict[str, Any], *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any) -> Any:
        self.is_responding = False
        self.response = outputs['response']
        end_time = time.time()  # Capture the end time
        latency = end_time - self.start_time  # Calculate latency

        # Log the question, response, and latency
        self.log_interaction(self.question, self.response, latency)

        # Save the interaction and embeddings into the database after generating the response
        # user_input_plus_response = f"{self.original_question}\n\n{self.response}"
        # embeddings = self.memory_db.get_local_embedding(user_input_plus_response)
        # self.memory_db.save_to_database(user_input_plus_response, embeddings)

        print("END: "+self.response)

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