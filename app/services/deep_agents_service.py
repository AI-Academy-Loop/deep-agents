import os
from deepagents import create_deep_agent
from config import settings

research_instructions = """You are an expert researcher. Your job is to conduct thorough research, and then write a polished report.
"""

from langchain.chat_models import init_chat_model
 
llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai",api_key=settings.GOOGLE_API_KEY)

agent = create_deep_agent(
    model=llm,
    system_prompt=research_instructions,
)

