from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from config import settings
from openai import OpenAI

app = FastAPI()

class Query(BaseModel):
    text: str

# Создаем базовый клиент OpenAI
client = OpenAI(
    api_key=settings.BOTHUB_API_KEY,
    base_url=settings.BOTHUB_API_BASE
)

# Используем его в LangChain
llm = ChatOpenAI(
    model_name=settings.MODEL_NAME,
    temperature=settings.TEMPERATURE,
    client=client  # передаем готовый клиент
)

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
    Эндпоинт для отправки запросов к BotHub через Langchain
    """
    try:
        response = await chain.ainvoke({"text": query.text})
        return {
            "response": response,
            "status": "success"
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
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
