import chromadb
from openai import OpenAI
import os
import uuid
from chromadb.config import Settings


class rag:
    def __init__(self):
        self.embed_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.embed_model = "text-embedding-3-small"
        self.client = chromadb.CloudClient(
        api_key=os.getenv("CHROMA_API_KEY"),
        tenant=os.getenv("CHROMA_TENANT"),
        database=os.getenv("CHROMA_DATABASE")
        )

    def embed_texts(self, texts):
            if isinstance(texts, str):
                 texts = [texts]
            response = self.embed_client.embeddings.create(
                model=self.embed_model,
                input=texts
            )

            return [item.embedding for item in response.data]
        
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
    

            