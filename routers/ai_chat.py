from fastapi import APIRouter, HTTPException, Request, Depends,Header
from pydantic import BaseModel, Field
from core.dependencies import get_llm
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    prompt: str
    

@router.post("/ask")
def ask(request: Request, body: ChatRequest,x_api_key: str = Header(...), llm=Depends(get_llm)):
    logger.info("POST /ask called")
    logger.debug(f"Client: {request.client}")
    logger.debug(f"Collection (api_key): {x_api_key}")

    if not body.prompt:
        logger.warning("Empty prompt received")
        raise HTTPException(status_code=400, detail="No prompt provided")

    if not llm.collection_exists(x_api_key):
        logger.warning(f"Collection not found: {x_api_key}")
        raise HTTPException(status_code=404, detail="Collection not found")

    try:
        response = llm.ask_llm(body.prompt, x_api_key)
        logger.info("LLM response generated successfully")
        return {"response": response}

    except Exception:
        logger.exception("Error during LLM processing")
        raise HTTPException(status_code=500, detail="Internal server error")