import json
import os
import sqlite3

import numpy as np
import pandas as pd
from llama_cpp import Llama
from sklearn.metrics.pairwise import cosine_similarity

"""
    Handles all long/short term memory via SQL Lite
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
            embeddings_json = json.dumps(embeddings)  # Convert embeddings to a JSON string for storage
            cursor.execute('''
                INSERT INTO chat_history (user_input_plus_response, embeddings)
                VALUES (?, ?)
            ''', (user_input_plus_response, embeddings_json))  # Insert data into the table
            conn.commit()
        finally:
            conn.close()


    def load_to_dataframe(self):

        if not os.path.exists(self.db_path):
            return pd.DataFrame()  # Return an empty DataFrame if the database doesn't exist
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

        # Convert embeddings from JSON strings back to lists for similarity calculations
        df['embeddings'] = df['embeddings'].apply(lambda emb: json.loads(emb))

        # Compute similarity for each row by comparing embeddings with the current input embedding
        df['similarity'] = df['embeddings'].apply(lambda emb: self.similarity_search(user_input_embedding, emb))

        # Sort by similarity score and return the top-n most relevant records
        top_n_df = df.sort_values(by='similarity', ascending=False).head(n)

        return top_n_df


    @staticmethod
    def similarity_search(embedding1, embedding2):
        embedding1 = np.array(embedding1).reshape(1, -1)
        embedding2 = np.array(embedding2).reshape(1, -1)
        return cosine_similarity(embedding1, embedding2)[0][0]


    def get_local_embedding(self, text):
        embedding = self.embedding_model.create_embedding(text)['data'][0]['embedding']
        return embedding
