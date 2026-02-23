from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

router = APIRouter()

def get_llm(app):
    if not hasattr(app.state, "llm"):
        from core.util_instances import get_llm_instance
        app.state.llm = get_llm_instance()
    return app.state.llm


class ChatRequest(BaseModel):
    prompt: str
    api_key: str = Field(alias="x-api-key")


@router.post("/ask")
def ask(request: Request, body: ChatRequest):
    llm_instance = get_llm(request.app)

    if not llm_instance.collection_exists(body.api_key):
        raise HTTPException(status_code=404, detail="Collection not found")

    response = llm_instance.ask_llm(body.prompt, body.api_key)
    return {"response": response}