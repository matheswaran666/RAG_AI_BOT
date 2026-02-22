from fastapi import APIRouter,File,UploadFile,HTTPException
from pydantic import BaseModel, Field
from core.util_instances import llm_instance,utils_instance,rag_instance



router = APIRouter(prefix="/create_collection")



@router.post("/create_collection")
async def ask(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    docs = await file.read()
    print(docs)
    docs = docs.decode("utf-8").split("\n")
    print(docs)
    if not docs:
        raise HTTPException(status_code=400, detail="File is empty")

    api_key = utils_instance.generate_api_key()
    response = llm_instance.create_rag_collection(api_key,docs)

    return {"response": response, "x-api-key": api_key}