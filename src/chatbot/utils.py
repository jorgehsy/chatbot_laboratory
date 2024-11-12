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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PriceRange(BaseModel):
    min_price: float
    max_price: float


# Decorators
def async_retry(retries: int = 3, delay: float = 1.0):
    """Retry async function on failure"""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                    if attempt < retries - 1:
                        await asyncio.sleep(delay * (attempt + 1))
            raise last_exception

        return wrapper

    return decorator


def measure_time(func):
    """Measure execution time of a function"""

    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"Function {func.__name__} took {duration:.2f} seconds to execute")
        return result

    return wrapper


# Database utilities
def sanitize_input(value: str) -> str:
    """Sanitize input strings for database operations"""
    if not isinstance(value, str):
        return value
    # Remove SQL injection patterns
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
    """Safely execute database operations with proper error handling"""
    try:
        return await operation
    except SQLAlchemyError as e:
        logger.error(f"Database error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise


# Price and currency utilities
def format_price(amount: Union[float, Decimal], currency: str = "USD") -> str:
    """Format price with proper currency symbol and decimals"""
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
    """Calculate discount based on quantity tiers"""
    discount_percentage = 0
    for tier_quantity, tier_discount in sorted(discount_tiers.items(), reverse=True):
        if quantity >= tier_quantity:
            discount_percentage = tier_discount
            break

    return total_amount * (1 - discount_percentage / 100)


# Validation utilities
def validate_phone_number(phone: str) -> bool:
    """Validate phone number format"""
    phone_pattern = re.compile(r'^\+?1?\d{9,15}$')
    return bool(phone_pattern.match(phone))


def validate_email(email: str) -> bool:
    """Validate email format"""
    email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    return bool(email_pattern.match(email))


def validate_address(address: str) -> bool:
    """Validate shipping address"""
    # Basic address validation
    if not address or len(address.strip()) < 10:
        return False
    required_elements = [
        r'\d+',  # Street number
        r'[A-Za-z\s]+',  # Street name
        r'[A-Za-z\s]+,',  # City
        r'[A-Za-z]{2,}'  # State or Country
    ]
    return all(re.search(pattern, address) for pattern in required_elements)


# Date and time utilities
def get_utc_timestamp() -> datetime:
    """Get current UTC timestamp"""
    return datetime.now(timezone.utc)


def format_datetime(dt: datetime, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Format datetime object to string"""
    return dt.strftime(format_str)


def parse_datetime(dt_str: str, format_str: str = "%Y-%m-%d %H:%M:%S") -> datetime:
    """Parse datetime string to datetime object"""
    return datetime.strptime(dt_str, format_str)


# Order processing utilities
class OrderValidator:
    @staticmethod
    def validate_order_items(items: List[Dict]) -> bool:
        """Validate order items structure"""
        required_fields = {'product_id', 'quantity'}
        return all(
            all(field in item for field in required_fields) and
            isinstance(item['quantity'], int) and
            item['quantity'] > 0
            for item in items
        )

    @staticmethod
    def validate_order_context(context: Dict) -> bool:
        """Validate complete order context"""
        required_fields = {
            'customer_id',
            'items',
            'shipping_address'
        }
        return all(field in context for field in required_fields)


# Text processing utilities
def extract_product_info(text: str) -> Dict[str, Any]:
    """Extract product information from text"""
    # Basic pattern matching for product information
    quantity_pattern = r'(\d+)\s*(?:units?|pieces?|pcs?)'
    product_pattern = r'(?i)(?:need|want|order|buy)\s+(?:\d+\s+)?(.+?)(?=\s*for|$|\s*at\s)'

    quantity_match = re.search(quantity_pattern, text)
    product_match = re.search(product_pattern, text)

    return {
        'quantity': int(quantity_match.group(1)) if quantity_match else None,
        'product': product_match.group(1).strip() if product_match else None
    }


def extract_price_range(text: str) -> Optional[PriceRange]:
    """Extract price range from text"""
    # Pattern for price ranges like "$100-$200" or "between $100 and $200"
    pattern = r'\$(\d+(?:\.\d{2})?)\s*(?:to|-|and)\s*\$(\d+(?:\.\d{2})?)'
    match = re.search(pattern, text)

    if match:
        return PriceRange(
            min_price=float(match.group(1)),
            max_price=float(match.group(2))
        )
    return None


# Cache utilities
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


# Error handling utilities
class ChatbotError(Exception):
    """Base exception for chatbot errors"""
    pass


class ValidationError(ChatbotError):
    """Validation error in chatbot operations"""
    pass


class DatabaseError(ChatbotError):
    """Database operation error"""
    pass


def handle_error(error: Exception) -> Dict[str, str]:
    """Convert exceptions to user-friendly messages"""
    error_messages = {
        ValidationError: "Invalid input provided",
        DatabaseError: "Unable to process request",
        SQLAlchemyError: "Database operation failed",
        ValueError: "Invalid value provided",
    }

    error_type = type(error)
    message = error_messages.get(error_type, "An unexpected error occurred")

    logger.error(f"Error occurred: {str(error)}")
    return {
        "error": message,
        "detail": str(error),
        "type": error_type.__name__
    }


# JSON utilities
def safe_json_loads(json_str: str) -> Dict:
    """Safely load JSON string"""
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {str(e)}")
        return {}


def safe_json_dumps(obj: Any) -> str:
    """Safely dump object to JSON string"""
    try:
        return json.dumps(obj, default=str)
    except (TypeError, ValueError) as e:
        logger.error(f"JSON encode error: {str(e)}")
        return "{}"

def parse_openai_content(content) -> ...:
    return content.replace('```json', '').replace('```', '').strip()


# Conversation context utilities
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


# Export all utilities
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