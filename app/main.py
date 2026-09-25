from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from app.routers import upload, chat

# Load environment variables
load_dotenv()

app = FastAPI(title="Clarity")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "https://clarrity.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(upload.router)
app.include_router(chat.router)
from app.routers.agent_chat import router as agent_router
app.include_router(agent_router)
from app.routers.documents import router as documents_router
app.include_router(documents_router)
from app.routers.agent_stream import router as stream_router
app.include_router(stream_router)
from app.routers.eval import router as eval_router
app.include_router(eval_router)

@app.get("/")
async def root():
    return {"message": "Welcome to Clarity API"}

import traceback
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version="1.0.0",
        routes=app.routes,
    )

    def _fix_schema(s):
        if isinstance(s, dict):
            if s.get("contentMediaType") == "application/octet-stream":
                del s["contentMediaType"]
                s["format"] = "binary"
            for v in s.values():
                _fix_schema(v)
        elif isinstance(s, list):
            for item in s:
                _fix_schema(item)

    _fix_schema(openapi_schema)
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    traceback.print_exc()
    return JSONResponse(status_code=500, content = {"detail" : str(exc)})

