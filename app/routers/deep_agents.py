from fastapi import APIRouter
from services.deep_agents_service import agent

router = APIRouter()

@router.post("/deep-agents")
async def get_deep_agents(query: str):
    response = agent.invoke({"messages": [{"role": "user", "content": query}]})
    return response
