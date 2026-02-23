from fastapi import APIRouter, File, UploadFile, HTTPException, Request, Depends
from pydantic import BaseModel, Field
from core.dependencies import get_llm

router = APIRouter()




class api_key(BaseModel):
    api_key: str = Field(alias="x-api-key")

@router.post("/audio_to_ask")
def audio_to_ask(request: Request, body: api_key, file: UploadFile = File(...),llm = Depends(get_llm)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    text = llm.audio_to_text(file)
    response = llm.ask_llm(text, body.api_key)
    return {"response": response}
