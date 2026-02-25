import chromadb
from google import genai
import os
import uuid
from chromadb.config import Settings
import requests
from uuid import uuid4

class rag:
    def __init__(self):
        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY4"))      
        self.vectorDB = chromadb.CloudClient(
        api_key=os.getenv("CHROMA_API_KEY"),
        tenant=os.getenv("CHROMA_TENANT"),
        database=os.getenv("CHROMA_DATABASE")
        )
        self.api_key_num = 4

    def change_api_key(self):
        self.api_key_num += 1
        self.client = genai.Client(api_key="GOOGLE_API_KEY"+str(self.api_key_num))


    def embed_texts(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        response = self.client.models.embed_content(
            model="models/gemini-embedding-001",
            contents=texts
        )


        return [e.values for e in response.embeddings]     
        

    def create_collections(self, collection_name, docs):

            if isinstance(docs, str):
                docs = [docs]

            collection = self.get_or_create_collection(collection_name)

            embeddings = self.embed_texts(docs)

            ids = [str(uuid4()) for _ in docs]

            collection.add(
                documents=docs,
                embeddings=embeddings,
                ids=ids
            )

            return "collection created successfully"
    
    
    def delete_doc_from_collection(self,collection_name,docId):
        collection = self.vectorDB.get_collection(collection_name)

    def get_or_create_collection(self, collection_name):
        try:
            return self.vectorDB.get_collection(collection_name)
        except:
            return self.vectorDB.create_collection(collection_name)


    def search_docs(self,query,collection_name):
        query = self.embed_texts([query])
        results = self.vectorDB.get_collection(collection_name).query(
            query_embeddings=query, 
            n_results=2
        )
        retrieved_text = "\n".join(results["documents"][0])
        return retrieved_text
    
    def collection_exists(self,collection_name):
        try:
            self.vectorDB.get_collection(collection_name)
            return True
        except:
            return False
    

            