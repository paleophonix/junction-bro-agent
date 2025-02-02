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
regex_template = """Ты - эксперт по регулярным выражениям в Java 8. Создай точный regex-шаблон для поиска и замены текста, используя ТОЛЬКО синтаксис Java 8 regex.

Важные правила Java 8 regex:
- Используй $1, $2 и т.д. для обратных ссылок в replacement (не \1, \2)
- Для начала строки используй \A вместо ^
- Для конца строки используй \z вместо $
- Для экранирования используй \Q...\E
- Для флагов используй (?i), (?m), (?s)
- Для позитивного просмотра вперед используй (?=...)
- Для негативного просмотра вперед используй (?!...)
- Для позитивного просмотра назад используй (?<=...)
- Для негативного просмотра назад используй (?<!...)

ВАЖНО: В шаблоне замены (replacement) используй ТОЛЬКО:
- $1, $2, и т.д. для обратных ссылок на группы
- Обычный текст и пробелы
- Знаки пунктуации
НЕ ИСПОЛЬЗУЙ никакие Java-функции или методы в шаблоне замены!

Инструкция: {instruction}

Пример преобразования:
До: {text_before}
После: {text_after}

История предыдущих запросов:
{history}

ВАЖНО: Ответь ТОЛЬКО двумя строками в точном формате:
Pattern: <шаблон>
Replacement: <замена>

Не добавляй никаких пояснений, кавычек или дополнительного текста."""

# Добавляем новый шаблон для самооценки
evaluation_template = """Оцени результат применения regex-шаблона в Java-нотации.

Исходная задача:
Инструкция: {instruction}
Текст до замены: {text_before}
Ожидаемый результат: {text_after}

Использованные шаблоны:
Pattern: {pattern}
Replacement: {replacement}

Фактический результат: {actual_result}

Пожалуйста, оцени результат по шкале от 1 до 10, где:
   1-3: Полное несоответствие ожиданиям
   4-6: Частичное соответствие
   7-9: Хороший результат с мелкими недочетами
   10: Полное соответствие

Формат ответа:
Score: (число от 1 до 10)
Evaluation: (краткое описание того, что произошло)
"""

# Добавляем новый шаблон для анализа и планирования
analysis_template = """Проанализируй неудачную попытку создания regex-шаблона в Java-нотации и предложи план исправления. Будь краток, не предлагай программирование.


Пожалуйста, предоставь:
1. Анализ причин несоответствия
2. Конкретный план по улучшению шаблона

Формат ответа:
Analysis: (анализ проблемы)
Plan: (план исправления)
Исходные данные:
Инструкция: {instruction}
Ожидаемый результат: {text_after}

Текущая попытка:
Pattern: {pattern}
Replacement: {replacement}
Результат: {actual_result}

История предыдущих попыток:
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
            input_variables=["instruction", "text_after", "pattern", "replacement", "actual_result"],
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

    async def process_request(self, request: RegexRequest) -> dict:
        MAX_ATTEMPTS = 100
        
        try:
            attempt = 0
            best_score = 0
            best_response = None
            
            while attempt < MAX_ATTEMPTS:
                attempt += 1
                
                # Получаем паттерн и замену
                memory = self.memory.get_relevant_history(request)
                pattern, replacement = await self._get_pattern_and_replacement(request, memory)
                
                # Применяем regex и проверяем результат
                test_result = await self._apply_regex(pattern, replacement, request.text_before)
                
                # Оцениваем результат
                evaluation_response = await self._evaluate_result(request, pattern, replacement, test_result)
                score = self._parse_score(evaluation_response)
                
                # Нормализуем результаты для сравнения
                normalized_test = self._normalize_text(test_result)
                normalized_expected = self._normalize_text(request.text_after)
                
                # Логируем попытку
                self.session_logger.info(f"\nAttempt {attempt} - Assistant output:")
                self.session_logger.info(f"Pattern: {pattern!r}")
                self.session_logger.info(f"Replacement: {replacement!r}")
                self.session_logger.info(f"Result:\n{test_result}")
                
                # Сохраняем попытку в память
                attempt_info = f"""Attempt {attempt}:
Pattern: {pattern}
Replacement: {replacement}
Score: {score}
Evaluation: {evaluation_response}
"""
                self.memory.save_context(
                    {"input": request.text_before},
                    {"output": attempt_info}
                )
                
                # Если нашли точное совпадение - возвращаем его
                if normalized_test == normalized_expected:
                    self.session_logger.info(f"\nFound exact match on attempt {attempt}")
                    response = {
                        "pattern": pattern,
                        "replacement": replacement,
                        "test_result": test_result,
                        "self_evaluation": evaluation_response,
                        "score": score
                    }
                    return response
                
                # Сохраняем лучший результат
                if score > best_score:
                    best_score = score
                    best_response = {
                        "pattern": pattern,
                        "replacement": replacement,
                        "test_result": test_result,
                        "self_evaluation": evaluation_response,
                        "score": score
                    }
                
                # Если не нашли точное совпадение, пробуем проанализировать и улучшить
                if attempt < MAX_ATTEMPTS:
                    analysis = await self._analyze_result(request, pattern, replacement, test_result)
                    if analysis:
                        self.session_logger.info(f"\nAnalysis:\n{analysis}")
                
                if attempt >= MAX_ATTEMPTS:
                    self.session_logger.info(f"\nMax attempts ({MAX_ATTEMPTS}) reached")
                    return best_response or {
                        "pattern": pattern,
                        "replacement": replacement,
                        "test_result": test_result,
                        "self_evaluation": "Failed to find exact match",
                        "score": 1
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
            
            logger.info(f"Cleaned response: {response}")
            
            pattern = None
            replacement = None
            
            # Ищем Pattern: и Replacement: в тексте
            pattern_match = re.search(r'Pattern:\s*(.*?)(?=\n|$)', response)
            replacement_match = re.search(r'Replacement:\s*(.*?)(?=\n|$)', response)
            
            if pattern_match:
                pattern = pattern_match.group(1).strip()
                # Убираем двойное экранирование
                pattern = pattern.replace('\\\\', '\\')
            if replacement_match:
                replacement = replacement_match.group(1).strip()
                # Убираем только явные комментарии
                if ' Note:' in replacement:
                    replacement = replacement.split(' Note:')[0].strip()
            
            if not pattern or not replacement:
                raise ValueError(f"Pattern или Replacement не найдены в ответе")
            
            logger.info(f"Found pattern: '{pattern}'")
            logger.info(f"Found replacement: '{replacement}'")
            
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

    async def _apply_regex(self, pattern: str, replacement: str, text: str) -> str:
        """Применяет регулярное выражение к тексту"""
        try:
            # Очищаем шаблон от кавычек
            pattern = pattern.strip('`')
            replacement = replacement.strip('`')
            
            # Компилируем регулярное выражение
            regex = re.compile(pattern, re.MULTILINE | re.DOTALL)
            
            # Получаем все совпадения
            matches = list(regex.finditer(text))
            if not matches:
                self.logger.warning("No matches found!")
                return text
            
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
            return text

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
                text_before=request.text_before,
                text_after=request.text_after,
                pattern=pattern,
                replacement=replacement,
                actual_result=test_result
            )
            
            # Используем ainvoke и возвращаем строку
            response = await self.llm.ainvoke(prompt)
            if hasattr(response, 'content'):
                response = response.content
            return str(response)
            
        except Exception as e:
            self.logger.error(f"Error analyzing result: {str(e)}")
            raise

    async def _get_pattern_and_replacement(self, request: RegexRequest, memory: str) -> tuple[str, str]:
        """Получает паттерн и замену из ответа модели"""
        try:
            prompt = self.prompt.format(
                instruction=request.instruction,
                text_before=request.text_before,
                text_after=request.text_after,
                history=memory
            )
            
            # Получаем ответ от модели
            response = await self.llm.ainvoke(prompt)
            
            # Извлекаем content из ответа
            if hasattr(response, 'content'):
                response = response.content
            response = str(response)
            
            # Очищаем от метаданных
            if "content='" in response:
                response = response.split("content='")[1].split("' additional_kwargs")[0]
            
            self.logger.info(f"Raw model response: {response!r}")
            
            # Ищем Pattern: и Replacement: в тексте
            pattern_match = re.search(r'Pattern:\s*(.*?)(?=\nReplacement:|$)', response)
            replacement_match = re.search(r'Replacement:\s*(.*?)(?=\n|$)', response)
            
            if not pattern_match or not replacement_match:
                raise ValueError(f"Failed to extract pattern or replacement from response: {response}")
            
            pattern = pattern_match.group(1).strip()
            replacement = replacement_match.group(1).strip()
            
            self.logger.info(f"Extracted pattern: {pattern!r}")
            self.logger.info(f"Extracted replacement: {replacement!r}")
            
            return pattern, replacement
            
        except Exception as e:
            self.logger.error(f"Error getting pattern and replacement: {str(e)}")
            raise

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
        # Используем ensure_ascii=False чтобы отключить экранирование
        json_str = json.dumps(result, ensure_ascii=False)
        # Преобразуем обратно в dict
        clean_result = json.loads(json_str)
        return JSONResponse(content=clean_result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 