from fastapi import FastAPI, HTTPException
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from pydantic import BaseModel
from config import settings

# Инициализация OpenAI с использованием ключа из config
llm = ChatOpenAI(temperature=0.7, openai_api_key=settings.OPENAI_API_KEY)

# Инициализация FastAPI приложения
app = FastAPI(title="LLM Agent API")

# Создаем базовую модель для запроса
class Query(BaseModel):
    text: str
    
# Создаем промпт для общения
prompt = PromptTemplate(
    input_variables=["query"],
    template="""Ты - полезный ассистент. 
    Пожалуйста, ответь на следующий вопрос максимально информативно: {query}"""
)

# Создаем цепочку для обработки запросов
chain = LLMChain(llm=llm, prompt=prompt)

# Определяем инструменты для агента
tools = [
    Tool(
        name="Language Model",
        func=chain.run,
        description="Полезно для ответов на общие вопросы и анализа"
    )
]

# Инициализируем агента
agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
    verbose=True
)


@app.post("/ask")
async def ask_agent(query: Query):
    """
    Эндпоинт для отправки запросов к LLM-агенту
    """
    try:
        response = agent.run(query.text)
        return {"response": response, "status": "success"}
    except Exception as e:
        return {"error": str(e), "status": "error"}

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
