from fastapi import APIRouter, File, UploadFile, HTTPException, Request

router = APIRouter()

def get_llm(app):
    if not hasattr(app.state, "llm"):
        from core.util_instances import get_llm_instance
        app.state.llm = get_llm_instance()
    return app.state.llm

def get_utils(app):
    if not hasattr(app.state, "utils"):
        from core.util_instances import get_utils_instance
        app.state.utils = get_utils_instance()
    return app.state.utils


@router.post("/create_collection")
async def ask(request: Request, file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")

    docs = await file.read()
    docs = docs.decode("utf-8").split("\n")

    if not docs:
        raise HTTPException(status_code=400, detail="File is empty")

    llm = get_llm(request.app)
    utils = get_utils(request.app)

    api_key = utils.generate_api_key()
    response = llm.create_rag_collection(api_key, docs)

    return {"response": response, "x-api-key": api_key}