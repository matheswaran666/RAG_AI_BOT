from fastapi import APIRouter,HTTPException
from pydantic import BaseModel, Field
from core.util_instances import llm_instance

router = APIRouter(prefix="/ai_chat")

class chat_request(BaseModel):
    prompt: str
    api_key: str = Field(alias="x-api-key")



@router.post("/ask")
def ask(request: chat_request):
    if not llm_instance.collection_exists(request.api_key):
        raise HTTPException(status_code=404, detail="Collection not found")
    response = llm_instance.ask_llm(request.prompt, request.api_key)
    return {"response": response}



