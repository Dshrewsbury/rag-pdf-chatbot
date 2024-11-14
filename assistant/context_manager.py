from llama_cpp import Llama
from qdrant_client import QdrantClient

from assistant.memory_database import MemoryDatabaseManager


class ContextManager:
    def __init__(self):
        self.embedding_llm = Llama(
            model_path="../models/mxbai-embed-large-v1-f16.gguf",
            embedding=True,
            verbose=False,
            n_batch=512,
            max_tokens=512
        )
        self.memory_db = MemoryDatabaseManager(db_path="../chat_history.db", embedding_llm=self.embedding_llm)
        self.client = QdrantClient(path="../embeddings/recursive")


    def build_context(self, user_key: str, query: str) -> (str, str):
        # Fetch relevant documents based on the query
        query_vector = self.embedding_llm.create_embedding(query)['data'][0]['embedding']
        search_results = self.client.search(
            collection_name="recursive",
            query_vector=query_vector,
            limit=1
        )

        context = "\n\n".join([row.payload['text'] for row in search_results])

        # Retrieve user-specific conversation history
        user_input_embedding = self.memory_db.get_local_embedding(query)
        top_similar = self.memory_db.find_top_n_similar(user_input_embedding, n=1)
        relevant_history = ''
        if not top_similar.empty:
            relevant_history = "\n".join(top_similar['user_input_plus_response'].values)
            # print(f"Similar past interactions found:\n{similar_results}\n")

        return context, relevant_history


    @staticmethod
    def build_prompt(context: str, history: str, query: str) -> str:
        """
        Constructs the final prompt by combining context, history, and query.
        """
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

        # Structure the prompt
        instruction = f"Recent Chat History:\n\n{history} \n\nUser Question: {query}"
        system_prompt = B_SYS + """
        You are a helpful assistant on the Llama 2 paper. Answer the user's questions related to the paper using the provided context if relevant.
        Separate context from the question, answer directly, and avoid unnecessary details.
        """ + E_SYS

        return f"{B_INST}{system_prompt}Context: {context}\n\n{instruction}{E_INST}\nAI:"


    def update_memory(self, user_key: str, query: str, response: str):
        """
        Updates the user's memory with the latest conversation data.
        """
        user_input_plus_response = f"{query}\n\n{response}"
        embeddings = self.memory_db.get_local_embedding(user_input_plus_response)
        self.memory_db.save_to_database(user_input_plus_response, embeddings)
