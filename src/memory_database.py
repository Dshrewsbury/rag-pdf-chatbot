# memory_database.py
import sqlite3
import json
import numpy as np
import pandas as pd
import os
from llama_cpp import Llama
from sklearn.metrics.pairwise import cosine_similarity
"""
    SQLLite implementation of long term memory management
    Using a basic similarity search
    Currently storing the combined quetion + response
"""
class MemoryDatabaseManager:
    def __init__(self, db_path: str, embedding_llm: Llama):
        self.db_path = db_path
        self.embedding_model = embedding_llm
        self.setup_database()

    def setup_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_input_plus_response TEXT,
                embeddings TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def save_to_database(self, user_input_plus_response, embeddings):
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            embeddings_json = json.dumps(embeddings)  # Convert embeddings to JSON string
            cursor.execute('''
                INSERT INTO chat_history (user_input_plus_response, embeddings)
                VALUES (?, ?)
            ''', (user_input_plus_response, embeddings_json))
            conn.commit()
        finally:
            conn.close()

    def load_to_dataframe(self):
        if not os.path.exists(self.db_path):
            return pd.DataFrame()
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM chat_history", conn)
        conn.close()
        return df

    def find_top_n_similar(self, user_input_embedding, n=3):
        df = self.load_to_dataframe()

        if df.empty:
            print("The DataFrame is empty. No similarity search can be performed.")
            return pd.DataFrame()

        if 'embeddings' not in df.columns:
            print("The DataFrame does not contain an 'embeddings' column.")
            return pd.DataFrame()

        # Convert embeddings from JSON strings back to Python objects
        df['embeddings'] = df['embeddings'].apply(lambda emb: json.loads(emb))

        # Compute similarity for each row in the DataFrame
        df['similarity'] = df['embeddings'].apply(lambda emb: self.similarity_search(user_input_embedding, emb))

        # Sort the DataFrame based on similarity and return the top N results
        top_n_df = df.sort_values(by='similarity', ascending=False).head(n)

        return top_n_df

    def similarity_search(self, embedding1, embedding2):
        # Ensure the embeddings are in the correct shape (2D arrays)
        embedding1 = np.array(embedding1).reshape(1, -1)
        embedding2 = np.array(embedding2).reshape(1, -1)
        return cosine_similarity(embedding1, embedding2)[0][0]

    def get_local_embedding(self, text):
        embedding = self.embedding_model.create_embedding(text)['data'][0]['embedding']
        return embedding