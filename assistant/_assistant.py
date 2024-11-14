import hashlib
import uuid
from typing import Any, Dict, Optional

from llama_cpp import Llama

from assistant.context_manager import ContextManager
from assistant.streaming_handler import StreamingCallbackHandler

"""
Generic RAG AI Assistant defines the overall architecture and methods needed for managing assistant sessions. 
"""


class Assistant:
    llm: Llama


    def __init__(self, model_path: str):
        self.llm = Llama(
            verbose=True,
            model_path=model_path,
            n_batch=256,
            n_ctx=1024,
            top_k=100,
            top_p=0.37,
            temperature=0.8,
            max_tokens=200,
        )
        self.context_manager = ContextManager()
        self.session_data: Dict[str, Dict[str, Any]] = {}


    def add_session(self, key: str, **kwargs: Any):

        hashed_key = hashlib.sha256(key.encode()).hexdigest()

        if hashed_key not in self.session_data:
            self.session_data[hashed_key] = {
                "context": None,
                "history": [],
                "handler": StreamingCallbackHandler()
            }

        return self.session_data[hashed_key], hashed_key


    def get_session(self, hashed_key: str) -> Optional[Dict[str, Any]]:
        return self.session_data.get(hashed_key)


    def invoke_chain(self, user_key: str, query: str) -> str:

        # Retrieve or create user session data
        response_id = uuid.uuid4()
        session_data, hashed_key = self.add_session(user_key)
        handler = session_data["handler"]
        handler.on_start(question=query, response_id=response_id)

        # Fetch context and memory for the prompt
        context, history = self.context_manager.build_context(user_key, query)

        # Update the session data with the latest context and history
        session_data["context"] = context
        session_data["history"] = history

        # Build prompt with context, history, and query
        prompt = self.context_manager.build_prompt(context=context, history=history, query=query)

        # Generate response from the model
        stream = self.llm(
            prompt,
            max_tokens=100,
            stream=True,
        )

        response = ""
        for output in stream:
            token = output["choices"][0]["text"]  # Replace with actual tokenized generation
            handler.on_token(token)  # Stream each token
            response += token

        handler.on_end(response)

        # Update memory with the new conversation turn
        self.context_manager.update_memory(user_key, query, response)

        return response
