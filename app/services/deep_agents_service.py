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
    """Search ICD-10-CM codes for diagnosis. Returns: code, description, chapter/section context, includes/excludes notes. 
    Example: query='acute asthma attack' returns codes like J45.901 with full details."""
    if not parsing.cm_vectorstore: return "No CM data loaded."
    docs = parsing.cm_vectorstore.similarity_search(query, k=top_k)
    results = []
    for d in docs:
        # Extract code and description from content
        content_lines = d.page_content.split('\n')
        code = d.metadata.get('code', 'N/A')
        desc = d.metadata.get('description', '')
        if not desc and len(content_lines) > 1:
            # Try to extract from content
            for line in content_lines:
                if line.startswith('Description:'):
                    desc = line.replace('Description:', '').strip()
                    break
        
        results.append({
            "code": code,
            "description": desc,
            "details": d.page_content[:300],
            "source": d.metadata.get('source'),
            "type": d.metadata.get('type')
        })
    return json.dumps(results, indent=2)

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
    """Search ICD-10-PCS procedure codes. Returns: procedure tables, code definitions, approach/device options.
    Example: query='knee replacement' returns relevant PCS tables with code combinations."""
    if not parsing.pcs_vectorstore: return "No PCS data."
    docs = parsing.pcs_vectorstore.similarity_search(query, k=top_k)
    results = []
    for d in docs:
        results.append({
            "table_code": d.metadata.get('table_code', ''),
            "details": d.page_content[:300],
            "source": d.metadata.get('source'),
            "type": d.metadata.get('type')
        })
    return json.dumps(results, indent=2)

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
    #interrupt_on={
    #    "search_cm_codes": True,
    #    "extract_cm_guidelines": True,
    #    "search_pcs_codes": True
    #},
    checkpointer=checkpointer,
    system_prompt=system_prompt
)
