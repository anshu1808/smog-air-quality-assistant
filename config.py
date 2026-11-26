import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv(override=True)

groq_api_key = os.getenv("GROQ_API_KEY", "").strip()

if not groq_api_key:
    raise RuntimeError("GROQ_API_KEY is missing.")

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=groq_api_key
)
