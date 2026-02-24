import os
from dotenv import load_dotenv
from .rag import rag
from google import genai
import io
import requests
load_dotenv()
import json



class llm:
    def __init__(self):
        print("model loading...")
        print("LLM INIT — PID:", os.getpid())

        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY4"))
        self.rag = rag()
        self.model = "gemini-3-flash-preview"     
        # self.asr = pipeline(
        #     "automatic-speech-recognition",
        #     model="openai/whisper-small",
        #     generate_kwargs={
        #         "task": "transcribe",
        #         "language": "en"
        #     }
        # )

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
        return self.rag.create_collections(collection_name,response)


    def add_doc_to_collection(self,collection_name,docs,metadata=None):
        self.rag.add_doc_to_collection(collection_name,docs,metadata)

    
    def audio_to_text(self, audio_bytes):
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
            return response.text

        

    def ask_llm(self, prompt, collection_name):

            retrieved_text = self.rag.search_docs(prompt, collection_name)
            if not retrieved_text:
                return "I don't have enough information in the collection."
            prompt = f""" 
                {prompt} 
                context : {retrieved_text}
                instruction : 
                    Return only a valid JSON object.The first character of your response must be "{".The last character of your response must be "}". you include anything else, the response is invalid 

                        Do not wrap it in markdown.
                        Do not include ``` or ```json.
                        Do not include explanations.
                        Do not include any text before or after the JSON.


                            """

            return self.generate_response(prompt)
        
    def generate_response(self,prompt):
        response_obj = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
             config={
                "response_mime_type": "application/json"
            }
        )
        response = response_obj.text
        return response

    
    def collection_exists(self,collection_name):
        return self.rag.collection_exists(collection_name)
    

