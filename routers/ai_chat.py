from fastapi import APIRouter, HTTPException, Request, Depends
from pydantic import BaseModel, Field
from core.dependencies import get_llm


router = APIRouter()

class ChatRequest(BaseModel):
    prompt: str
    api_key: str = Field(alias="x-api-key")


@router.post("/ask")
def ask(request: Request, body: ChatRequest , llm = Depends(get_llm)):
    print("ask chat")
    if not body.prompt:
        raise HTTPException(status_code=400, detail="No prompt provided")
        
    if not llm.collection_exists(body.api_key):
        raise HTTPException(status_code=404, detail="Collection not found")
        
    response = llm.ask_llm(body.prompt, body.api_key)
    return {"response": response}