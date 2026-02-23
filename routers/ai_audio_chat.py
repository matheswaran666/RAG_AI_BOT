from fastapi import APIRouter, File, UploadFile, HTTPException, Request, Depends
from pydantic import BaseModel, Field

router = APIRouter()


def get_llm(app):
    if not hasattr(app.state, "llm"):
        from core.util_instances import get_llm_instance
        app.state.llm = get_llm_instance()
    return app.state.llm


class api_key(BaseModel):
    api_key: str = Field(alias="x-api-key")

@router.post("/audio_to_ask")
def audio_to_ask(request: Request, body: api_key, file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    llm = get_llm(request.app)
    text = llm.audio_to_text(file)
    response = llm.ask_llm(text, body.api_key)
    return {"response": response}
