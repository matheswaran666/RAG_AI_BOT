import os
from dotenv import load_dotenv
from openai import OpenAI
from .rag import rag
from transformers import pipeline
from google import genai
import torch
import io

torch.set_num_threads(8)
load_dotenv()


class llm:
    def __init__(self):
        print("model loading...")
        print("LLM INIT — PID:", os.getpid())

        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY1"))
        self.rag = rag()
        self.model = "gemini-2.5-flash"     
        self.asr = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-small",
            generate_kwargs={
                "task": "transcribe",
                "language": "en"
            }
        )

               

    def create_rag_collection(self,collection_name,docs):
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
            "contents":[],
            "response_structure": {
                "format_type": "",
                "fields": [],
                "structure":{}
            },
            "subject": "",
            "difficulty_levels": []
            }

            Extraction rules:

            - "allowed_tasks" should include only tasks explicitly mentioned (e.g., question_generation, hint_generation, performance_analysis).
            - "rules" must be direct constraints stated in the document.
            - "contents" every contents and informations should be included.
            - "response_structure" must describe required output format if specified.
            - "difficulty_levels" must include only explicitly stated levels.
            - If something is not explicitly written, return null or empty array.
            - Do not assume missing information.

            Document:
            """ + ("\n".join(docs) if isinstance(docs, list) else docs)
                    }]
                }
            ]

        response = self.generate_response(contents)
        self.rag.add_doc_to_collection(collection_name,response)

    def add_doc_to_collection(self,collection_name,docs,metadata=None):
        self.rag.add_doc_to_collection(collection_name,docs,metadata)

    def audio_to_text(self, file_path):
        print(file_path)
        result = self.asr(file_path)
        print(result["text"])   
        return result["text"]
    
   

    def ask_llm(self, prompt, collection_name):

            retrieved_text = self.rag.search_docs(prompt, collection_name)
            if not retrieved_text:
                return "I don't have enough information in the collection."
            prompt = f""" 
                {prompt} 
                context : {retrieved_text}
                            """

            return self.generate_response(prompt)
        
    def generate_response(self,prompt):
        response_obj = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
        )
        response = response_obj.text
        return response
    
    def collection_exists(self,collection_name):
        return self.rag.collection_exists(collection_name)
    

