from fastapi import APIRouter
from services.deep_agents_service import agent

router = APIRouter()

@router.post("/deep-agents")
async def get_deep_agents(query: str):
    response = await agent.ainvoke({"messages": [{"role": "user", "content": query}]})
    return {"response": response}