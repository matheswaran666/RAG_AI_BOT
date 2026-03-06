import chromadb
from google import genai
import os
from uuid import uuid4
import logging

logger = logging.getLogger(__name__)


class rag:
    def __init__(self):
        logger.info("Initializing RAG system...")
        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY1"))

        self.vectorDB = chromadb.CloudClient(
            api_key=os.getenv("CHROMA_API_KEY"),
            tenant=os.getenv("CHROMA_TENANT"),
            database=os.getenv("CHROMA_DATABASE")
        )

        self.api_key_num = 1

    def change_api_key(self):
        self.api_key_num += 1
        if self.api_key_num > 16:
            self.api_key_num = 1

        logger.warning(f"Switching embedding API key to #{self.api_key_num}")

        self.client = genai.Client(
            api_key=os.getenv("GOOGLE_API_KEY" + str(self.api_key_num))
        )

    def embed_texts(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        logger.debug(f"Embedding {len(texts)} text(s)")

        try:
            response = self.client.models.embed_content(
                model="models/gemini-embedding-001",
                contents=texts
            )
            return [e.values for e in response.embeddings]

        except Exception:
            logger.exception("Embedding failed. changing API key.")
            return self.embed_texts(texts)

    def create_collections(self, collection_name, docs):
        logger.info(f"Creating collection: {collection_name}")

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

        logger.debug(f"Added {len(ids)} document(s) to collection")
        return "collection created successfully"

    def delete_doc_from_collection(self, collection_name, doc_id):
        logger.info(f"Deleting document {doc_id} from {collection_name}")

        collection = self.vectorDB.get_collection(collection_name)
        collection.delete(ids=[doc_id])

        return "Document deleted successfully"

    def get_or_create_collection(self, collection_name):
        try:
            logger.debug(f"Fetching collection: {collection_name}")
            return self.vectorDB.get_collection(collection_name)
        except Exception:
            logger.warning(f"Collection not found. Creating: {collection_name}")
            return self.vectorDB.create_collection(collection_name)

    def search_docs(self, query, collection_name):
        logger.debug(f"Searching collection {collection_name}")

        query_embedding = self.embed_texts(query)

        results = self.vectorDB.get_collection(collection_name).query(
            query_embeddings=query_embedding,
            n_results=2
        )

        if not results["documents"]:
            logger.warning("No documents retrieved.")
            return ""

        retrieved_text = "\n".join(results["documents"][0])
        return retrieved_text

    def collection_exists(self, collection_name):
        try:
            self.vectorDB.get_collection(collection_name)
            return True
        except Exception:
            return False

    def get_docs_from_collection(self, collection_name):
        logger.debug(f"Getting documents from {collection_name}")
        return self.vectorDB.get_collection(collection_name).get()

    def add_doc_to_collection(self, collection_name, docs, metadata=None):
        collection = self.get_or_create_collection(collection_name)
        logger.info(f"Adding document to {collection_name}")
        if isinstance(docs, str):
            docs = [docs]
        collection.add(
            ids=[str(uuid4()) for _ in docs],
            documents=docs,
            metadatas=metadata,
            embeddings=self.embed_texts(docs)
        )

    def update_docs_in_collection(
        self,
        collection_name,
        ids,
        documents=None,
        metadatas=None,
        embeddings=None
    ):
        logger.info(f"Updating documents in {collection_name}")

        collection = self.vectorDB.get_collection(collection_name)

        collection.update(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )

        return "Documents updated successfully"

    def document_exists_in_collection(self, collection_name, doc_id):
        logger.debug(f"Checking if document exists: {doc_id}")

        collection = self.vectorDB.get_collection(collection_name)
        result = collection.get(ids=[doc_id])

        return len(result["ids"]) > 0