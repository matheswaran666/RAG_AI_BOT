from fastapi import APIRouter, HTTPException, Request, Depends
from pydantic import BaseModel, Field
from core.dependencies import get_llm
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    prompt: str
    api_key: str = Field(alias="x-api-key")


@router.post("/ask")
def ask(request: Request, body: ChatRequest, llm=Depends(get_llm)):
    logger.info("POST /ask called")
    logger.debug(f"Client: {request.client}")
    logger.debug(f"Collection (api_key): {body.api_key}")

    if not body.prompt:
        logger.warning("Empty prompt received")
        raise HTTPException(status_code=400, detail="No prompt provided")

    if not llm.collection_exists(body.api_key):
        logger.warning(f"Collection not found: {body.api_key}")
        raise HTTPException(status_code=404, detail="Collection not found")

    try:
        response = llm.ask_llm(body.prompt, body.api_key)
        logger.info("LLM response generated successfully")
        return {"response": response}

    except Exception:
        logger.exception("Error during LLM processing")
        raise HTTPException(status_code=500, detail="Internal server error")