import sys
import time
from typing import List, Optional
from uuid import UUID

"""
Handles real-time token streaming for each user query.
Manages session information like response tracking and latency logging.
"""


class StreamingCallbackHandler:

    def __init__(self):
        self.tokens: List[str] = []
        self.is_responding: bool = False
        self.response_id: Optional[UUID] = None
        self.response: Optional[str] = None
        self.start_time: Optional[float] = None
        self.question = ""


    def on_token(self, token: str):
        sys.stdout.write(token)
        sys.stdout.flush()
        self.tokens.append(token)


    def on_start(self, question: str, response_id: UUID):
        self.is_responding = True
        self.response_id = response_id
        self.question = question
        self.start_time = time.time()
        self.tokens.clear()


    def on_end(self, response: str):
        # Log the question, response, and latency
        self.is_responding = False
        self.response = response
        end_time = time.time()
        latency = end_time - self.start_time if self.start_time else 0
        self.log_interaction(self.question, self.response, latency)

        # Clear
        self.question = ""
        self.response = ""


    def get_response(self) -> str:
        response_result = self.response or ""
        self.response = None  # Clear after retrieval
        return response_result


    @staticmethod
    def log_interaction(question: str, response: str, latency: float):
        with open("chat_log.txt", "a") as f:
            f.write(f"Question: {question}\n")
            f.write(f"Response: {response}\n")
            f.write(f"Latency: {latency:.2f} seconds\n\n")
