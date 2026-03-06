from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from core.dependencies import get_llm, get_utils
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any



router = APIRouter()

@router.post("/create_collection")
async def create_collection(file: UploadFile = File(...),llm = Depends(get_llm),utils = Depends(get_utils)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")

    docs = await file.read()
    docs = docs.decode("utf-8").split("\n")

    if not docs:
        raise HTTPException(status_code=400, detail="File is empty")

    api_key = utils.generate_api_key()
    response = llm.create_rag_collection(api_key, docs)

    return {"response": response, "x-api-key": api_key}


class GetDocsModel(BaseModel):
    api_key: str = Field(alias="x-api-key")

@router.post("/get_docs")
def get_docs(body:GetDocsModel,llm = Depends(get_llm)):
    if not llm.collection_exists(body.api_key):
        raise HTTPException(status_code=404, detail="Collection not found")
    
    return llm.get_docs_from_collection(body.api_key)



class DocumentUpdate(BaseModel):
    doc_id: str
    document: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class UpdateDocsModel(BaseModel):
    api_key: str = Field(alias="x-api-key")
    docs: List[DocumentUpdate]  


class addDocsModel(BaseModel):
    api_key : str = Field(alias="x-api-key")
    docs : List[object]


@router.post("/update_docs_in_collection")
def update_docs_in_collection(body:UpdateDocsModel,llm = Depends(get_llm)):
    if not llm.collection_exists(body.api_key):
        raise HTTPException(status_code=404, detail="Collection not found")
    
    if not llm.docs_exists_in_collection(body.api_key , body.doc_id.doc_id):
        raise HTTPException(status_code=404, detail="Document not found")
    
    return llm.update_docs_in_collection(api_key,body.docs.doc_id,body.docs.document,body.docs.metadata)



@router.post("/add_doc_to_collection")
def add_doc_to_collection(body:addDocsModel,llm = Depends(get_llm)):
    if not llm.collection_exists(body.api_key):
        raise HTTPException(status_code=404, detail="Collection not found")
    
    if not body.docs:
        raise HTTPException(status_code=400, detail="No documents provided")

    return llm.add_doc_to_collection(body.api_key,str(body.docs))