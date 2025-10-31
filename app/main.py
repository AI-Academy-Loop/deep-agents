from fastapi import FastAPI
from app.routers.files import router as files_router
from app.utils.parsing import load_local_files
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load ICD-10 files into vector stores
    print("\n🚀 Loading ICD-10 data on startup...")
    load_local_files()
    yield
    # Shutdown: cleanup if needed
    print("\n👋 Shutting down...")

app = FastAPI(lifespan=lifespan)

app.include_router(files_router)


@app.get("/")
async def root():
    return {"message": "Hello World"}