from deepagents import create_deep_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from app.config import settings
from typing import Optional
import json
from langchain_chroma import Chroma
import app.utils.parsing as parsing

llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai",api_key=settings.GOOGLE_API_KEY)

@tool
def search_cm_codes(query: str, top_k: int = 5) -> str:
    """Semantic search CM Index/Tabular for diagnosis."""
    if not parsing.cm_vectorstore: return "No CM data loaded."
    docs = parsing.cm_vectorstore.similarity_search(query, k=top_k)
    return json.dumps([{
        "code": d.metadata.get('code', 'N/A'),
        "snippet": d.page_content[:200],
        "source": d.metadata.get('source')
    } for d in docs], indent=2)

@tool
def extract_cm_guidelines(query: str, top_k: int = 3) -> str:
    """Extract guidelines matching query."""
    if not parsing.cm_vectorstore: return "No guidelines."
    docs = parsing.cm_vectorstore.similarity_search(query, k=top_k, filter={"type": "guidelines"})
    return '\n'.join([d.page_content for d in docs])

@tool
def search_pcs_codes(query: str, top_k: int = 5) -> str:
    """Search PCS procedures."""
    if not parsing.pcs_vectorstore: return "No PCS data."
    docs = parsing.pcs_vectorstore.similarity_search(query, k=top_k)
    return json.dumps([{"snippet": d.page_content[:200], "source": d.metadata['source']} for d in docs])

@tool
def hybrid_search(query: str) -> str:
    """Parallel: CM codes + guidelines + PCS."""
    return f"Hybrid results for '{query}': Use other tools for details."

@tool
def get_stats() -> str:
    """RAG stats."""
    return f"CM chunks: {parsing.cm_vectorstore._collection.count() if parsing.cm_vectorstore else 0}, PCS: {parsing.pcs_vectorstore._collection.count() if parsing.pcs_vectorstore else 0}"

def get_config(thread_id: str):
    return {"configurable": {"thread_id": thread_id}}

# A checkpointer is required for the agent to be able to pause and resume.
checkpointer = MemorySaver()

system_prompt = """ICD-10 Expert Agent.

For DIAGNOSIS input:
1. hybrid_search to overview.
2. search_cm_codes(query) for codes.
3. extract_cm_guidelines(query) for rules.
4. If procedures: search_pcs_codes.
5. Rank + explain.
6. Output STRICT JSON:
{
  "codes": [{"code": "J45.901", "desc": "...", "reason": "...", "confidence": 0.95}],
  "guidelines": ["text1", "text2"],
  "pcs_codes": [...] | null,
  "explanation": "brief"
}
Interrupt for HITL on suggestions."""

agent = create_deep_agent(
    model=llm,
    tools=[search_cm_codes, extract_cm_guidelines, search_pcs_codes, hybrid_search, get_stats],
    interrupt_on={
        "search_cm_codes": True,
        "extract_cm_guidelines": True,
        "search_pcs_codes": True
    },
    checkpointer=checkpointer,
    system_prompt=system_prompt
)
