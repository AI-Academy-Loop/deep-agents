from fastapi import FastAPI
from app.routers.deep_agents import router as deep_agents_router

app = FastAPI()

app.include_router(deep_agents_router)

@app.get("/")
async def root():
    return {"message": "Hello World"}