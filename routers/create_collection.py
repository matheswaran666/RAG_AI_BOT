from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from core.dependencies import get_llm, get_utils

router = APIRouter()

@router.post("/create_collection")
async def ask(file: UploadFile = File(...),llm = Depends(get_llm),utils = Depends(get_utils)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")

    docs = await file.read()
    docs = docs.decode("utf-8").split("\n")

    if not docs:
        raise HTTPException(status_code=400, detail="File is empty")

    api_key = utils.generate_api_key()
    response = llm.create_rag_collection(api_key, docs)

    return {"response": response, "x-api-key": api_key}