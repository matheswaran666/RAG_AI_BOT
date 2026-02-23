from fastapi import FastAPI
from routers import ai_chat, create_collection, ai_audio_chat
import os

app = FastAPI()

app.include_router(ai_chat.router, prefix="/ai_chat")
app.include_router(ai_audio_chat.router, prefix="/ai_chat")
app.include_router(create_collection.router, prefix="/create_collection")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)