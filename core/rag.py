import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os
import uuid
from chromadb.config import Settings


class rag:
    def __init__(self):
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.client = chromadb.CloudClient(
        api_key=os.getenv("CHROMA_API_KEY"),
        tenant=os.getenv("CHROMA_TENANT"),
        database=os.getenv("CHROMA_DATABASE")
        )

    
    def create_collections(self,collection_name,docs):
        collection = self.get_or_create_collection(collection_name)
        embeddings = self.embed_model.encode(docs).tolist()
        collection.add(
        documents=docs,
        embeddings=embeddings,
        ids=[str(uuid.uuid4()) for _ in range(len(docs))]
        )
    
    def add_doc_to_collection(self,collection_name,docs,metadata=None):
        print("add_doc_to_collection_")
        collection = self.get_or_create_collection(collection_name)
        embedding = self.embed_model.encode(docs).tolist()
        collection.add(
            documents=docs,
            embeddings=embedding,
            ids = str(uuid.uuid4()),
            metadatas=metadata if metadata else None
        )
    
    def delete_doc_from_collection(self,collection_name,docId):
        collection = self.client.get_collection(collection_name)

    def get_or_create_collection(self, collection_name):
        try:
            return self.client.get_collection(collection_name)
        except:
            return self.client.create_collection(collection_name)

    def search_docs(self,query,collection_name):
        query = self.embed_model.encode([query]).tolist()
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
    

            