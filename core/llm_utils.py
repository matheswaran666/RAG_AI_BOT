import os
import json
from dotenv import load_dotenv
from .rag import rag
from google import genai
from google.genai import errors
import logging
import time

load_dotenv()

logger = logging.getLogger(__name__)


class llm:
    def __init__(self):
        logger.info("Model loading...")
        logger.debug(f"LLM INIT — PID: {os.getpid()}")

        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY1"))
        self.rag = rag()
        self.model = os.getenv("MODEL1")
        self.model_num = 1
        self.api_key_num = 1
        self.request_count = 0
        self.available_api_keys_count = 17
        self.available_models_keys_count = 4

    def create_rag_collection(self, collection_name, docs):
        logger.info(f"Creating RAG collection: {collection_name}")

        contents = [
            {
                "role": "user",
                "parts": [{
                    "text": """
You are an information extraction engine.
Your task is to convert educational configuration documents into structured JSON.
You must return strictly valid JSON.
Do not include explanations.
Do not include markdown.
Do not include commentary.
If information is missing, return null.
Do not invent information that is not explicitly stated.

Analyze the following document and extract structured configuration using this exact schema:

{
  "allowed_tasks": [],
  "rules": [],
  "contents": [],
  "response_structure": {
    "format_type": "",
    "fields": [],
    "structure": {}
  },
  "subject": "",
  "difficulty_levels": []
}

Extraction rules:

- "allowed_tasks" should include only tasks explicitly mentioned.
- "rules" must be direct constraints stated in the document.
- "contents" every contents and informations should be included.
- "response_structure" must describe required output format if specified.
- "difficulty_levels" must include only explicitly stated levels.
- If something is not explicitly written, return null or empty array.
- Do not assume missing information.

Document:
"""
                    + ("\n".join(docs) if isinstance(docs, list) else docs)
                }]
            }
        ]

        response = self.generate_response(contents)
        logger.debug("RAG collection created successfully.")
        return self.rag.create_collections(collection_name, response)

    def add_doc_to_collection(self, collection_name, docs, metadata=None):
        logger.info(f"Adding document to collection: {collection_name}")
        self.rag.add_doc_to_collection(collection_name, docs, metadata)

    def audio_to_text(self, audio_bytes):
        logger.info("Transcribing audio...")
        response = self.client.models.generate_content(
            model="gemini-1.5-flash",
            contents=[
                {
                    "role": "user",
                    "parts": [
                        {"mime_type": "audio/wav", "data": audio_bytes},
                        {"text": "Transcribe this audio."}
                    ]
                }
            ]
        )
        logger.debug("Audio transcription completed.")

        return response.text

    def change_api_key(self):
        self.api_key_num += 1
        if self.api_key_num > self.available_api_keys_count:
            self.api_key_num = 1
        # if self.api_key_num == self.available_api_keys_count - 5 and self.model_num == 1:
        #     self.switch_model()
        logger.warning(f"Switching to API key #{self.api_key_num}")

        self.client = genai.Client(
            api_key=os.getenv("GOOGLE_API_KEY" + str(self.api_key_num))
        )
    
    def switch_model(self):
        logger.info("Switching model...")
        self.model_num += 1
        if self.model_num > self.available_models_keys_count:
            self.model_num = 1

        self.model = os.getenv("MODEL" + str(self.model_num))
        logger.debug(f"Model switched to: {self.model}")

    
    def generate_response(self, prompt):
        logger.info("Calls count"+str(self.request_count))
        self.request_count += 1
        # Prevent hitting the 15 RPM limit on free tier
        if self.request_count % 10 == 0:
            logger.info("Cooling down for 60s...")
            time.sleep(60)
        
        try:
            logger.debug(f"Generating response using {self.model}...")
            
            response_obj = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={
                    "response_mime_type": "application/json"
                }
            )
            
            logger.debug("LLM response received successfully.")
            print(f"Model: {self.model} | Key Index: {self.api_key_num}")
            return response_obj.text

        except errors.ServerError:
            # Server-side overload (503)
            logger.warning("Model unavailable (503). Switching model...")
            self.switch_model()
            return self.generate_response(prompt)

        except errors.ClientError:
            # Rate limit or quota hit (429)
            logger.warning("Quota exceeded (429). Switching API key...")
            self.change_api_key()
            return self.generate_response(prompt)

        except Exception as e:
            logger.exception(f"Unexpected error occurred: {type(e)} - {e}")
            raise
    
    def ask_llm(self, prompt, collection_name):
        logger.info(f"LLM request for collection: {collection_name}")
        logger.debug(f"Prompt: {prompt}")

        retrieved_text = self.rag.search_docs(prompt, collection_name)

        if not retrieved_text:
            logger.warning("No retrieved context found.")
            return "I don't have enough information in the collection."

        prompt = f"""
            {prompt}
            context : {retrieved_text}
            instruction :
            Return only a valid JSON object.The first character of your response must be "{{".The last character of your response must be "}}". you include anything else, the response is invalid

            Do not wrap it in markdown.
            Do not include ``` or ```json.
            Do not include explanations.
            Do not include any text before or after the JSON.
            """
        return self.generate_response(prompt)


    def collection_exists(self, collection_name):
        logger.debug(f"Checking if collection exists: {collection_name}")
        return self.rag.collection_exists(collection_name)

    def get_docs_from_collection(self, collection_name):
        logger.debug(f"Getting documents from collection: {collection_name}")
        return self.rag.get_docs_from_collection(collection_name)

    def docs_exists_in_collection(self, collection_name, docId):
        logger.debug(f"Checking if document exists: {docId}")
        return self.rag.document_exists_in_collection(collection_name, docId)

    def update_docs_in_collection(
        self, collection_name, ids, documents=None, metadatas=None, embeddings=None
    ):
        logger.info(f"Updating documents in collection: {collection_name}")
        return self.rag.update_docs_in_collection(
            collection_name, ids, documents, metadatas, embeddings
        )