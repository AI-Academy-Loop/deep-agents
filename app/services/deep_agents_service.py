from deepagents import create_deep_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from app.config import settings
from typing import Optional
import json
import app.utils.parsing as parsing
import os

# Configure LangSmith tracing
os.environ["LANGSMITH_TRACING_V2"] = settings.LANGSMITH_TRACING_V2
os.environ["LANGSMITH_API_KEY"] = settings.LANGSMITH_API_KEY
os.environ["LANGSMITH_PROJECT"] = settings.LANGSMITH_PROJECT
os.environ["LANGSMITH_ENDPOINT"] = settings.LANGSMITH_ENDPOINT

llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai",api_key=settings.GOOGLE_API_KEY)

@tool
def search_cm_codes(query: str, top_k: int = 5) -> str:
    """Search ICD-10-CM codes for diagnosis. Returns: list of codes only."""
    if not parsing.cm_vectorstore: return "[]"
    docs = parsing.cm_vectorstore.similarity_search(query, k=top_k)
    codes = []
    for d in docs:
        code = d.metadata.get('code', 'N/A')
        if code != 'N/A':
            codes.append(code)
    return json.dumps(codes)

@tool
def extract_cm_guidelines(query: str, top_k: int = 3) -> str:
    """Extract ICD-10-CM coding guidelines and instructions. Returns: official guidelines, coding rules, chapter-specific instructions.
    Example: query='diabetes' returns guidelines for diabetes coding sequencing."""
    if not parsing.cm_vectorstore: return "No guidelines."
    docs = parsing.cm_vectorstore.similarity_search(query, k=top_k, filter={"type": "guidelines"})
    if not docs:
        # Fallback: search without filter if no guidelines found
        docs = parsing.cm_vectorstore.similarity_search(query, k=top_k)
    return '\n\n---\n\n'.join([d.page_content for d in docs])

@tool
def search_pcs_codes(query: str, top_k: int = 5) -> str:
    """Search ICD-10-PCS procedure codes. Returns: list of codes only."""
    if not parsing.pcs_vectorstore: return "[]"
    docs = parsing.pcs_vectorstore.similarity_search(query, k=top_k)
    codes = []
    for d in docs:
        code = d.metadata.get('code', d.metadata.get('table_code', ''))
        if code:
            codes.append(code)
    return json.dumps(codes)

@tool
def extract_pcs_guidelines(query: str, top_k: int = 3) -> str:
    """Extract ICD-10-PCS coding guidelines and instructions. Returns: official guidelines, coding rules, section-specific instructions.
    Example: query='cardiac procedures' returns guidelines for cardiac procedure coding."""
    if not parsing.pcs_vectorstore: return "No PCS guidelines."
    docs = parsing.pcs_vectorstore.similarity_search(query, k=top_k, filter={"type": "guidelines"})
    if not docs:
        # Fallback: search without filter if no guidelines found
        docs = parsing.pcs_vectorstore.similarity_search(query, k=top_k)
    return '\n\n---\n\n'.join([d.page_content for d in docs])

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

For any input, use the appropriate tool to search for codes.
Return ONLY a JSON array of codes found, e.g.:
["J45.901", "I10", "E11.9"]
Do not include explanations, guidelines, or any other information.
"""

agent = create_deep_agent(
    model=llm,
    tools=[search_cm_codes, extract_cm_guidelines, search_pcs_codes, extract_pcs_guidelines, hybrid_search, get_stats],
#    interrupt_on={
#        "search_cm_codes": True,
#        "extract_cm_guidelines": True,
#        "search_pcs_codes": True,
#        "extract_pcs_guidelines": True
#    },
    checkpointer=checkpointer,
    system_prompt=system_prompt
)
