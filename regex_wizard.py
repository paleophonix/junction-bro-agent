from pathlib import Path
from langchain_core.memory import BaseMemory
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import re
from config import settings
from openai import OpenAI
import httpx
from httpx_socks import AsyncProxyTransport
import logging
import json
from datetime import datetime
import asyncio
from fastapi import HTTPException, APIRouter
from fastapi.responses import JSONResponse
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Настраиваем логирование
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('regex_sessions.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('regex_wizard')

# Существующие шаблоны - оставляем
regex_template = """You are a regex pattern creation assistant. Create patterns for simple text editor find & replace function that work in Java 8.

RULES:
1. Pattern must use Java 8 notation:
   - $1, $2 for group references (not \1, \2)
   - \A for start of line (not ^)
   - \z for end of line (not $)
   - (?i) for case-insensitive
   - Basic regex: (), [], *, +, ?, \d, \w, \s

2. Replacement can ONLY use:
   - Group numbers: $1, $2, $3
   - Plain text
   - Spaces and punctuation

IMPORTANT: Do not use any quotes or backticks in your response!
No JavaScript, no Python, no conditions, no functions - just simple find & replace.

Instruction: {instruction}

Example transformation:
Before: {text_before}
After: {text_after}

Previous attempts history:
{history}

Reply ONLY with:
Pattern: <pattern>
Replacement: <replacement>"""

# Добавляем новый шаблон для самооценки
evaluation_template = """Evaluate the regex pattern application in Java notation.

Original task:
Instruction: {instruction}
Text before: {text_before}
Expected result: {text_after}

Used patterns:
Pattern: {pattern}
Replacement: {replacement}

Actual result: {actual_result}

Please rate the result on a scale from 1 to 10, where:
   1-3: Complete mismatch
   4-6: Partial match
   7-9: Good result with minor issues
   10: Perfect match

Response format:
Score: (number from 1 to 10)
Evaluation: (brief description of what happened)
"""

# Добавляем новый шаблон для анализа и планирования
analysis_template = """Analyze the failed regex pattern attempt in Java notation and suggest an improvement plan. Be concise.

Please provide:
1. Analysis of mismatch reasons
2. Specific plan for pattern improvement

Response format:
Analysis: (problem analysis)
Plan: (improvement plan)

Input data:
Instruction: {instruction}
Expected result: {text_after}

Current attempt:
Pattern: {pattern}
Replacement: {replacement}
Result: {actual_result}

Previous attempts history:
{history}
"""

class RegexRequest(BaseModel):
    instruction: str
    text_before: str
    text_after: str

class RegexResponse(BaseModel):
    """Модель ответа с регуляркой"""
    pattern: str
    replacement: str
    test_result: str
    self_evaluation: str
    score: int

class RegexMemory(ConversationBufferWindowMemory):
    """Память для хранения истории регулярок"""
    
    def get_relevant_history(self, request: RegexRequest) -> str:
        """Получает релевантную историю для запроса"""
        messages = self.chat_memory.messages
        if not messages:
            return ""
            
        history = []
        for i in range(0, len(messages), 2):
            if i+1 < len(messages):
                history.append(f"Input: {messages[i].content}\nOutput: {messages[i+1].content}\n")
        
        return "\n".join(history)

class RegexAgent:
    def __init__(self, llm: ChatOpenAI = None, memory: BaseMemory = None, logger: logging.Logger = None):
        if llm is None:
            base_url = "https://bothub.chat/api/v2/openai/v1/"
            llm = ChatOpenAI(
                model_name=settings.MODEL_NAME,
                openai_api_base=base_url,
                openai_api_key=settings.BOTHUB_API_KEY
            )
        self.llm = llm
        
        # Создаем нашу кастомную память
        if memory is None:
            memory = RegexMemory(
                k=5,  # Хранить последние 5 обменов
                memory_key="history",
                return_messages=True,
                output_key="output",
                input_key="input"
            )
        self.memory = memory
        
        # Создаем логгер по умолчанию если не передан
        if logger is None:
            logger = logging.getLogger("regex_wizard")
            logger.setLevel(logging.INFO)
            if not logger.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
                logger.addHandler(handler)
        self.logger = logger
        self.session_logger = None
        
        # Обновляем промпт с инструкцией про экранирование
        self.prompt = PromptTemplate(
            input_variables=["instruction", "text_before", "text_after", "history"],
            template=regex_template
        )
        
        self.analysis_prompt = PromptTemplate(
            input_variables=["instruction", "text_after", "pattern", "replacement", "actual_result", "history"],
            template=analysis_template
        )
        
        self.evaluation_prompt = PromptTemplate(
            input_variables=["instruction", "text_before", "text_after", "pattern", "replacement", "actual_result"],
            template=evaluation_template
        )
        
        # Создаем директорию для логов сессий если её нет
        self.sessions_dir = Path("sessions")
        self.sessions_dir.mkdir(exist_ok=True)
        
        # Создаем новую сессию
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_log = self.sessions_dir / f"{self.session_id}.log"
        
        # Настраиваем файловый хендлер для логов сессии
        self.session_handler = logging.FileHandler(self.session_log)
        self.session_handler.setFormatter(logging.Formatter('%(message)s'))
        self.session_logger = logging.getLogger(f"session_{self.session_id}")
        self.session_logger.addHandler(self.session_handler)
        self.session_logger.setLevel(logging.INFO)
        
        self.logger.info(f"Started new session: {self.session_id}")
        
        # Создаем отдельную цепочку для оценки
        self.evaluation_chain = self.evaluation_prompt | self.llm
        
        # Создаем цепочку для анализа
        self.analysis_chain = self.analysis_prompt | self.llm
    
    def _get_completion(self, prompt: str) -> str:
        """Получение ответа от модели"""
        try:
            response = self.llm.invoke(prompt)
            return response
        except Exception as e:
            logger.error(f"Error getting completion: {str(e)}", exc_info=True)
            raise

    def _clean_result(self, result: dict) -> dict:
        """Очищает результат от лишних слешей и метаданных"""
        if 'pattern' in result:
            result['pattern'] = result['pattern'].replace('\\\\', '\\')
            
        if 'self_evaluation' in result:
            eval_match = re.search(r"content='(.*?)'", result['self_evaluation'])
            if eval_match:
                result['self_evaluation'] = eval_match.group(1).replace('\\n', '\n')
                
        return result
    
    async def _process_regex_request(self, request: RegexRequest) -> dict:
        """Внутренний метод обработки запроса"""
        pattern, replacement = await self._get_pattern_and_replacement(request)
        if not pattern or not replacement:
            return {"error": "Failed to generate pattern"}
            
        test_result = await self._test_pattern(pattern, replacement, request.text_before)
        analysis = await self._analyze_result(request, pattern, replacement, test_result)
        score = self._extract_score(analysis)
        
        return {
            "pattern": pattern,
            "replacement": replacement
        }
    
    async def process_request(self, request: RegexRequest) -> dict:
        """Обработка запроса на создание regex"""
        try:
            total_input_tokens = 0
            total_output_tokens = 0
            
            # 1. Первый вызов - генерация паттерна
            response = await self.llm.ainvoke(self.prompt.format(
                instruction=request.instruction,
                text_before=request.text_before,
                text_after=request.text_after,
                history=self.memory.get_relevant_history(request)
            ))
            
            # Суммируем токены с генерации
            if hasattr(response, 'usage'):
                total_input_tokens += response.usage.prompt_tokens
                total_output_tokens += response.usage.completion_tokens
            
            # Получаем паттерн и замену
            if hasattr(response, 'content'):
                response = response.content
            response = str(response)
            
            pattern_match = re.search(r"Pattern: (.*?)(?:\n|$)", response)
            replacement_match = re.search(r"Replacement: (.*?)(?:\n|$)", response)
            
            if not pattern_match or not replacement_match:
                return {"error": "Failed to generate pattern"}
            
            pattern = pattern_match.group(1).strip()
            replacement = replacement_match.group(1).strip()
            
            # 2. Второй вызов - тестирование
            test_result = await self._test_pattern(pattern, replacement, request.text_before)
            test_response = await self._evaluate_result(request, pattern, replacement, test_result)
            
            # Суммируем токены с тестирования
            if hasattr(test_response, 'usage'):
                total_input_tokens += test_response.usage.prompt_tokens
                total_output_tokens += test_response.usage.completion_tokens
            
            # 3. Третий вызов - анализ
            analysis_response = await self._analyze_result(request, pattern, replacement, test_result)
            
            # Суммируем токены с анализа
            if hasattr(analysis_response, 'usage'):
                total_input_tokens += analysis_response.usage.prompt_tokens
                total_output_tokens += analysis_response.usage.completion_tokens
            
            return {
                "pattern": pattern.replace('\\\\', '\\'),
                "replacement": replacement,
                "metadata": {
                    'input_tokens': total_input_tokens,
                    'output_tokens': total_output_tokens
                }
            }
        except Exception as e:
            self.logger.error(f"Error processing request: {str(e)}")
            raise

    def _parse_response(self, response: str) -> tuple[str, str]:
        """Извлекает шаблон и строку замены из ответа модели"""
        try:
            logger.info(f"Parsing response: {response}")
            
            # Очищаем от метаданных
            if 'content="' in response:
                response = response.split('content="')[1].split('"')[0]
            elif "content='" in response:
                response = response.split("content='")[1].split("'")[0]
            
            # Убираем экранированные кавычки и переносы строк
            response = response.replace('\\"', '"').replace('\\n', '\n')
            
            pattern = None
            replacement = None
            
            # Ищем Pattern: и Replacement: в тексте
            pattern_match = re.search(r'Pattern:\s*(.*?)(?=\n|$)', response)
            replacement_match = re.search(r'Replacement:\s*(.*?)(?=\n|$)', response)
            
            if pattern_match:
                pattern = pattern_match.group(1).strip('`')  # Убираем бэктики!
            if replacement_match:
                replacement = replacement_match.group(1).strip('`')  # Убираем бэктики!
            
            if not pattern or not replacement:
                raise ValueError(f"Pattern или Replacement не найдены в ответе")
            
            return pattern, replacement
            
        except Exception as e:
            logger.error(f"Parse error: {str(e)}")
            logger.error(f"Response: {response}")
            raise ValueError(f"Ошибка при разборе ответа: {str(e)}")
    
    def _normalize_text(self, text: str) -> str:
        """Нормализует текст для сравнения"""
        # Убираем все пробельные символы
        text = re.sub(r'\s+', ' ', text)
        # Убираем пробелы в начале и конце
        text = text.strip()
        # Приводим к нижнему регистру для регистронезависимого сравнения
        text = text.lower()
        return text

    async def _test_pattern(self, pattern: str, replacement: str, text_before: str) -> str:
        """Применяет регулярное выражение к тексту"""
        try:
            # Очищаем шаблон от кавычек
            pattern = pattern.strip('`')
            replacement = replacement.strip('`')
            
            # Компилируем регулярное выражение
            regex = re.compile(pattern, re.MULTILINE | re.DOTALL)
            
            # Получаем все совпадения
            matches = list(regex.finditer(text_before))
            if not matches:
                self.logger.warning("No matches found!")
                return text_before
            
            # Применяем замену для каждого совпадения
            result = []
            for match in matches:
                # Получаем группы
                groups = match.groups()
                # Создаем замену, подставляя значения групп
                repl = replacement
                for i, group in enumerate(groups, 1):
                    repl = repl.replace(f'${i}', group)
                result.append(repl)
            
            # Объединяем результаты
            final_result = '\n\n'.join(result)
            self.logger.info(f"Found {len(matches)} matches")
            self.logger.info(f"Groups from first match: {matches[0].groups() if matches else 'none'}")
            self.logger.info(f"Final result: {final_result}")
            
            return final_result
            
        except Exception as e:
            self.logger.error(f"Error applying regex: {str(e)}")
            return text_before

    async def _evaluate_result(self, request: RegexRequest, pattern: str, replacement: str, test_result: str) -> str:
        """Оценивает результат применения регулярки"""
        try:
            # Создаем промпт для оценки
            prompt = self.evaluation_prompt.format(
                instruction=request.instruction,
                text_before=request.text_before,
                text_after=request.text_after,
                pattern=pattern,
                replacement=replacement,
                actual_result=test_result
            )
            
            # Используем ainvoke вместо apredict
            response = await self.llm.ainvoke(prompt)
            return str(response)
            
        except Exception as e:
            self.logger.error(f"Error evaluating result: {str(e)}")
            raise

    def _parse_score(self, evaluation: str) -> int:
        """Извлекает числовую оценку из текста оценки"""
        try:
            # Ищем оценку в формате "Score: X" или просто число
            match = re.search(r'Score:\s*(\d+)', evaluation) or re.search(r'^(\d+)$', evaluation)
            if match:
                score = int(match.group(1))
                # Ограничиваем оценку диапазоном 1-10
                return max(1, min(10, score))
            return 1
        except Exception as e:
            self.logger.error(f"Error parsing score: {str(e)}")
            return 1

    def _is_valid_java_regex(self, pattern: str, replacement: str) -> bool:
        """Проверяет, что регулярка соответствует синтаксису Java 8"""
        # Убираем кавычки
        pattern = pattern.strip('`')
        replacement = replacement.strip('`')
        
        # В Java 8 слеш в replacement должен быть экранирован
        if '/' in replacement and not '\\/' in replacement:
            self.logger.info("In Java 8 replacement, slash must be escaped: use \\/ instead of /")
            return False
        
        # Проверяем только критичные для Java 8 конструкции
        invalid_patterns = [
            r'^(?!\A)', r'$(?!\z)',  # ^ и $ без \A и \z
            r'(?<',  # Именованные группы
            r'\k<',  # Ссылки на именованные группы
            r'\1', r'\2',  # Обратные ссылки в стиле Perl
        ]
        
        for invalid in invalid_patterns:
            if invalid in pattern:
                self.logger.info(f"Invalid Java 8 pattern construct: {invalid}")
                return False
        
        # В replacement разрешаем только:
        # - $1, $2, и т.д.
        # - Обычный текст и пробелы
        # - Базовые знаки пунктуации
        # - Экранированный слеш (\/)
        allowed_replacement = re.match(r'^[a-zA-Z0-9\s\[\]\\\/:$\.,!?-]+$', replacement)
        if not allowed_replacement:
            self.logger.info("Replacement contains invalid characters for Java 8")
            return False
        
        # Проверяем, что все $ используются правильно
        dollars = re.findall(r'\$\d+', replacement)
        if not dollars:
            self.logger.info("No group references in replacement")
            return False
        
        other_dollars = re.findall(r'\$(?!\d+)', replacement)
        if other_dollars:
            self.logger.info("Invalid $ usage in replacement")
            return False
        
        # Проверяем, что слеши экранированы правильно
        if '/' in replacement:
            slashes = replacement.count('/')
            escaped_slashes = replacement.count('\\/')
            if slashes != escaped_slashes:
                self.logger.info("All slashes in replacement must be escaped in Java 8")
                return False
        
        return True

    async def _analyze_result(self, request: RegexRequest, pattern: str, replacement: str, test_result: str) -> str:
        """Анализирует результат применения регулярки"""
        try:
            # Создаем промпт для анализа
            prompt = self.analysis_prompt.format(
                instruction=request.instruction,
                text_after=request.text_after,
                pattern=pattern,
                replacement=replacement,
                actual_result=test_result,
                history=self.memory.get_relevant_history(request)
            )
            
            # Используем ainvoke и возвращаем строку
            response = await self.llm.ainvoke(prompt)
            if hasattr(response, 'content'):
                response = response.content
            return str(response)
            
        except Exception as e:
            self.logger.error(f"Error analyzing result: {str(e)}")
            return ""

    async def _get_pattern_and_replacement(self, request: RegexRequest) -> tuple[str, str]:
        """Получает паттерн и замену от модели"""
        try:
            # Получаем историю из памяти
            memory = self.memory.get_relevant_history(request)
            
            # Получаем ответ от модели
            response = await self.llm.ainvoke(self.prompt.format(
                instruction=request.instruction,
                text_before=request.text_before,
                text_after=request.text_after,
                history=memory
            ))
            
            # Преобразуем в строку и убираем двойные слеши
            if hasattr(response, 'content'):
                response = response.content
            response = str(response).replace('\\\\', '\\')
            
            # Извлекаем паттерн и замену
            pattern_match = re.search(r"Pattern: (.*?)(?:\n|$)", response)
            replacement_match = re.search(r"Replacement: (.*?)(?:\n|$)", response)
            
            if not pattern_match or not replacement_match:
                return "", ""
            
            return pattern_match.group(1).strip(), replacement_match.group(1).strip()
            
        except Exception as e:
            self.logger.error(f"Error getting pattern and replacement: {str(e)}")
            return "", ""

    def __del__(self):
        # Закрываем файл лога при удалении объекта
        if hasattr(self, 'session_handler'):
            self.session_handler.close()

# Создаем единственный экземпляр агента
regex_agent = RegexAgent()

router = APIRouter()

@router.post("/regex")
async def process_regex(request: RegexRequest) -> JSONResponse:
    try:
        result = await regex_agent.process_request(request)
        
        # Очищаем self_evaluation от метаданных
        if 'self_evaluation' in result:
            eval_match = re.search(r"content='(.*?)'", result['self_evaluation'])
            if eval_match:
                result['self_evaluation'] = eval_match.group(1)
            
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 