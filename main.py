from fastapi import FastAPI
from routers import ai_chat,create_collection
import os


app = FastAPI()

app.include_router(ai_chat.router,prefix="/ai_chat")
app.include_router(create_collection.router,prefix="/create_collection")


@app.on_event("startup")
async def startup_event():
    from core.util_instances import get_llm_instance, get_utils_instance

    app.state.llm = get_llm_instance()
    app.state.utils = get_utils_instance()
    app.state.rag = app.state.llm.rag


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)