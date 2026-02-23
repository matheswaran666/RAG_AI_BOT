import chromadb
from openai import OpenAI
import os
import uuid
from chromadb.config import Settings
import requests
from sentence_transformers import SentenceTransformer
class rag:
    def __init__(self):
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")       
        self.client = chromadb.CloudClient(
        api_key=os.getenv("CHROMA_API_KEY"),
        tenant=os.getenv("CHROMA_TENANT"),
        database=os.getenv("CHROMA_DATABASE")
        )

        # self.api_key = os.getenv("QUBRID_API_KEY")
        # self.url = "https://api.qubrid.ai/v1/embeddings"



    def embed_texts(self, texts):
            if isinstance(texts, str):
                 texts = [texts]
            return self.embed_model.encode(texts).tolist()        
        
    def create_collections(self,collection_name,docs):
        collection = self.get_or_create_collection(collection_name)
        collection.add(
        documents=docs,
        embeddings=self.embed_texts(docs),
        ids=[str(uuid.uuid4()) for _ in range(len(docs))]
        )
        return "collection created succesfully"

    
    
    def delete_doc_from_collection(self,collection_name,docId):
        collection = self.client.get_collection(collection_name)

    def get_or_create_collection(self, collection_name):
        try:
            return self.client.get_collection(collection_name)
        except:
            return self.client.create_collection(collection_name)

    def search_docs(self,query,collection_name):
        query = self.embed_texts([query])
        results = self.client.get_collection(collection_name).query(
            query_embeddings=query, 
            n_results=2
        )
        retrieved_text = "\n".join(results["documents"][0])
        return retrieved_text
    
    def collection_exists(self,collection_name):
        try:
            self.client.get_collection(collection_name)
            return True
        except:
            return False
    

            