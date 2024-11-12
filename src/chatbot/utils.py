from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timezone
import json
import re
from decimal import Decimal
import logging
from functools import wraps
import time
from pydantic import BaseModel, ValidationError
import asyncio
from sqlalchemy.exc import SQLAlchemyError

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PriceRange(BaseModel):
    min_price: float
    max_price: float


# Decoradores
def async_retry(retries: int = 3, delay: float = 1.0):
    """Reintentar función asíncrona en caso de fallo"""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    logger.warning(f"Intento {attempt + 1} falló: {str(e)}")
                    if attempt < retries - 1:
                        await asyncio.sleep(delay * (attempt + 1))
            raise last_exception

        return wrapper

    return decorator


def measure_time(func):
    """Medir tiempo de ejecución de una función"""

    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"La función {func.__name__} tardó {duration:.2f} segundos en ejecutarse")
        return result

    return wrapper


# Utilidades de base de datos
def sanitize_input(value: str) -> str:
    """Sanitizar cadenas de entrada para operaciones de base de datos"""
    if not isinstance(value, str):
        return value
    # Eliminar patrones de inyección SQL
    dangerous_patterns = [
        r"--",
        r";",
        r"'",
        r"\"",
        r"\/\*",
        r"\*\/",
        r"xp_",
        r"exec"
    ]
    sanitized = value
    for pattern in dangerous_patterns:
        sanitized = re.sub(pattern, "", sanitized, flags=re.IGNORECASE)
    return sanitized


async def safe_database_operation(operation):
    """Ejecutar operaciones de base de datos de forma segura con manejo de errores apropiado"""
    try:
        return await operation
    except SQLAlchemyError as e:
        logger.error(f"Error de base de datos: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error inesperado: {str(e)}")
        raise


# Utilidades de precio y moneda
def format_price(amount: Union[float, Decimal], currency: str = "USD") -> str:
    """Formatear precio con símbolo de moneda y decimales apropiados"""
    currency_symbols = {
        "USD": "$",
        "EUR": "€",
        "GBP": "£",
        "JPY": "¥"
    }
    symbol = currency_symbols.get(currency, "$")

    if currency == "JPY":
        return f"{symbol}{int(amount):,}"
    return f"{symbol}{float(amount):,.2f}"


def calculate_bulk_discount(
        total_amount: float,
        quantity: int,
        discount_tiers: Dict[int, float]
) -> float:
    """Calcular descuento basado en niveles de cantidad"""
    discount_percentage = 0
    for tier_quantity, tier_discount in sorted(discount_tiers.items(), reverse=True):
        if quantity >= tier_quantity:
            discount_percentage = tier_discount
            break

    return total_amount * (1 - discount_percentage / 100)


# Utilidades de validación
def validate_phone_number(phone: str) -> bool:
    """Validar formato de número telefónico"""
    phone_pattern = re.compile(r'^\+?1?\d{9,15}$')
    return bool(phone_pattern.match(phone))


def validate_email(email: str) -> bool:
    """Validar formato de correo electrónico"""
    email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    return bool(email_pattern.match(email))


def validate_address(address: str) -> bool:
    """Validar dirección de envío"""
    if not address or len(address.strip()) < 10:
        return False
    required_elements = [
        r'\d+',  # Número de calle
        r'[A-Za-z\s]+',  # Nombre de calle
        r'[A-Za-z\s]+,',  # Ciudad
        r'[A-Za-z]{2,}'  # Estado o País
    ]
    return all(re.search(pattern, address) for pattern in required_elements)


# Utilidades de fecha y hora
def get_utc_timestamp() -> datetime:
    """Obtener marca de tiempo UTC actual"""
    return datetime.now(timezone.utc)


def format_datetime(dt: datetime, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Formatear objeto datetime a cadena"""
    return dt.strftime(format_str)


def parse_datetime(dt_str: str, format_str: str = "%Y-%m-%d %H:%M:%S") -> datetime:
    """Analizar cadena datetime a objeto datetime"""
    return datetime.strptime(dt_str, format_str)


# Utilidades de procesamiento de pedidos
class OrderValidator:
    @staticmethod
    def validate_order_items(items: List[Dict]) -> bool:
        """Validar estructura de items del pedido"""
        required_fields = {'product_id', 'quantity'}
        return all(
            all(field in item for field in required_fields) and
            isinstance(item['quantity'], int) and
            item['quantity'] > 0
            for item in items
        )

    @staticmethod
    def validate_order_context(context: Dict) -> bool:
        """Validar contexto completo del pedido"""
        required_fields = {
            'customer_id',
            'items',
            'shipping_address'
        }
        return all(field in context for field in required_fields)


# Utilidades de procesamiento de texto
def extract_product_info(text: str) -> Dict[str, Any]:
    """Extraer información de producto del texto"""
    quantity_pattern = r'(\d+)\s*(?:unidades?|piezas?|pzs?)'
    product_pattern = r'(?i)(?:necesito|quiero|ordenar|comprar)\s+(?:\d+\s+)?(.+?)(?=\s*por|$|\s*a\s)'

    quantity_match = re.search(quantity_pattern, text)
    product_match = re.search(product_pattern, text)

    return {
        'quantity': int(quantity_match.group(1)) if quantity_match else None,
        'product': product_match.group(1).strip() if product_match else None
    }


def extract_price_range(text: str) -> Optional[PriceRange]:
    """Extraer rango de precios del texto"""
    pattern = r'\$(\d+(?:\.\d{2})?)\s*(?:a|-|y)\s*\$(\d+(?:\.\d{2})?)'
    match = re.search(pattern, text)

    if match:
        return PriceRange(
            min_price=float(match.group(1)),
            max_price=float(match.group(2))
        )
    return None


# Utilidades de caché
class SimpleCache:
    def __init__(self, ttl: int = 3600):
        self.cache = {}
        self.ttl = ttl

    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            value, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return value
            del self.cache[key]
        return None

    def set(self, key: str, value: Any):
        self.cache[key] = (value, time.time())

    def clear(self):
        self.cache.clear()


# Utilidades de manejo de errores
class ChatbotError(Exception):
    """Excepción base para errores del chatbot"""
    pass


class ValidationError(ChatbotError):
    """Error de validación en operaciones del chatbot"""
    pass


class DatabaseError(ChatbotError):
    """Error de operación de base de datos"""
    pass


def handle_error(error: Exception) -> Dict[str, str]:
    """Convertir excepciones a mensajes amigables para el usuario"""
    error_messages = {
        ValidationError: "Entrada no válida proporcionada",
        DatabaseError: "No se puede procesar la solicitud",
        SQLAlchemyError: "Falló la operación de base de datos",
        ValueError: "Valor no válido proporcionado",
    }

    error_type = type(error)
    message = error_messages.get(error_type, "Ocurrió un error inesperado")

    logger.error(f"Ocurrió un error: {str(error)}")
    return {
        "error": message,
        "detail": str(error),
        "type": error_type.__name__
    }


# Utilidades JSON
def safe_json_loads(json_str: str) -> Dict:
    """Cargar cadena JSON de forma segura"""
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error(f"Error de decodificación JSON: {str(e)}")
        return {}


def safe_json_dumps(obj: Any) -> str:
    """Volcar objeto a cadena JSON de forma segura"""
    try:
        return json.dumps(obj, default=str)
    except (TypeError, ValueError) as e:
        logger.error(f"Error de codificación JSON: {str(e)}")
        return "{}"


def parse_openai_content(content) -> ...:
    return content.replace('```json', '').replace('```', '').strip()


# Utilidades de contexto de conversación
class ConversationContext:
    def __init__(self):
        self.context = {}
        self.timestamp = datetime.now()

    def update(self, key: str, value: Any):
        self.context[key] = value
        self.timestamp = datetime.now()

    def get(self, key: str, default: Any = None) -> Any:
        return self.context.get(key, default)

    def clear(self):
        self.context.clear()
        self.timestamp = datetime.now()

    def is_expired(self, ttl: int = 3600) -> bool:
        return (datetime.now() - self.timestamp).total_seconds() > ttl


# Exportar todas las utilidades
__all__ = [
    'async_retry',
    'measure_time',
    'sanitize_input',
    'safe_database_operation',
    'format_price',
    'calculate_bulk_discount',
    'validate_phone_number',
    'validate_email',
    'validate_address',
    'get_utc_timestamp',
    'format_datetime',
    'parse_datetime',
    'OrderValidator',
    'extract_product_info',
    'extract_price_range',
    'SimpleCache',
    'ChatbotError',
    'ValidationError',
    'DatabaseError',
    'handle_error',
    'safe_json_loads',
    'safe_json_dumps',
    'ConversationContext',
    'logger',
    'parse_openai_content'
]