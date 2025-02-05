from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from config import settings
from openai import OpenAI
import os
import httpx
from httpx_socks import AsyncProxyTransport
from dotenv import load_dotenv
from regex_wizard import regex_agent, RegexRequest
import re
from fastapi.responses import JSONResponse, Response
import json
from typing import Any
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

print(f"Proxy from env: {settings.SOCKS5_PROXY}")
print(f"LLM Provider: {settings.LLM_PROVIDER}")
print(f"OpenAI API Key set: {'OPENAI_API_KEY' in os.environ}")
print(f"Bothub API Key set: {'BOTHUB_API_KEY' in os.environ}")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Для тестов можно разрешить всем
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    text: str

class RegexResponse(BaseModel):
    pattern: str
    replacement: str

# Выбор LLM провайдера
if settings.LLM_PROVIDER == "openai":
    print("Using OpenAI configuration")
    transport = AsyncProxyTransport.from_url(settings.SOCKS5_PROXY)
    client = OpenAI(
        api_key=settings.OPENAI_API_KEY,
        http_client=httpx.AsyncClient(transport=transport)
    )
    print(f"OpenAI base URL: {client.base_url}")
else:
    print("Using Bothub configuration")
    client = OpenAI(
        api_key=settings.BOTHUB_API_KEY,
        base_url=settings.BOTHUB_API_BASE
    )
    print(f"Bothub base URL: {client.base_url}")

# Используем его в LangChain
llm = ChatOpenAI(
    model_name=settings.MODEL_NAME,
    temperature=settings.TEMPERATURE,
    client=client,
    openai_api_base=settings.BOTHUB_API_BASE if settings.LLM_PROVIDER == "bothub" else None  # явно указываем base_url
)
print(f"LLM configuration: {llm.client.base_url}")

# Создание промпта
prompt = PromptTemplate(
    input_variables=["text"],
    template="{text}"
)

# Создание цепочки
chain = prompt | llm

@app.post("/ask")
async def ask_endpoint(query: Query):
    """
    Эндпоинт для отправки запросов через Langchain
    """
    try:
        print(f"Making request to: {llm.client.base_url}")
        response = await chain.ainvoke({"text": query.text})
        return {
            "response": response,
            "status": "success",
            "api": settings.LLM_PROVIDER,
            "endpoint": llm.client.base_url
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """
    Корневой эндпоинт для проверки работоспособности API
    """
    return {
        "status": "active",
        "message": "LLM Agent API работает",
        "active_api": settings.LLM_PROVIDER,
        "version": "1.0.0"
    }

@app.post("/regex/replace")
async def regex_endpoint(request: RegexRequest):
    """Эндпоинт для создания regex-шаблонов"""
    try:
        full_result = await regex_agent.process_request(request)
        return RegexResponse(
            pattern=full_result["pattern"],
            replacement=full_result["replacement"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True  # включаем автоперезагрузку
    )
