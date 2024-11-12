from typing import Dict, List, Optional, Union
import json
from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field
from anthropic import Anthropic
import openai
import asyncio
from functools import lru_cache


class LLMProvider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)


class EntityExtraction(BaseModel):
    customer_name: Optional[str] = None
    customer_id: Optional[int] = None
    products: List[Dict[str, Union[str, int, float]]] = Field(default_factory=list)
    quantities: List[int] = Field(default_factory=list)
    shipping_address: Optional[str] = None
    special_instructions: Optional[str] = None


class Intent(BaseModel):
    primary: str
    confidence: float
    entities: EntityExtraction
    requires_clarification: bool = False
    suggested_next_state: Optional[str] = None


class LLMHandler:
    def __init__(self, provider: LLMProvider = LLMProvider.ANTHROPIC, model_config: Dict = None):
        self.provider = provider
        self.model_config = model_config or self._get_default_config()
        self.conversation_history: List[Message] = []

        if provider == LLMProvider.ANTHROPIC:
            self.client = Anthropic()
            self.default_model = "claude-3-opus-20240229"
        else:
            self.client = openai.OpenAI()
            self.default_model = "gpt-4o"

    def _get_default_config(self) -> Dict:
        """Get default configuration for the LLM"""
        return {
            LLMProvider.ANTHROPIC: {
                "temperature": 0.7,
                "max_tokens": 1000,
                "model": "claude-3-opus-20240229"
            },
            LLMProvider.OPENAI: {
                "temperature": 0.7,
                "max_tokens": 1000,
                "model": "gpt-4o"
            }
        }[self.provider]

    @lru_cache(maxsize=100)
    def _get_system_prompt(self, context: str = "") -> str:
        """Get cached system prompt with context"""
        return f"""You are an AI sales assistant helping with order processing.
        Your role is to understand customer requests and extract relevant information.

        Current Context:
        {context}

        Guidelines:
        1. Extract customer details, products, quantities, and shipping information
        2. Identify the primary intent of each message
        3. Maintain a professional and helpful tone
        4. Ask for clarification when needed
        5. Handle complex orders and special requests appropriately

        Available Actions:
        - Order processing
        - Product inquiries
        - Shipping information
        - Order modifications
        - Status checks
        """

    async def extract_intent(self, message: str, context: str = "") -> Intent:
        """Extract intent and entities from user message"""
        prompt = f"""
        Analyze the following message and extract order-related information.

        Message: "{message}"

        Return a JSON object with:
        {{
            "primary": "intent_type",
            "confidence": 0.0 to 1.0,
            "entities": {{
                "customer_name": "name or null",
                "customer_id": number or null,
                "products": [
                    {{"name": "product_name", "quantity": number}}
                ],
                "shipping_address": "address or null",
                "special_instructions": "instructions or null"
            }},
            "requires_clarification": boolean,
            "suggested_next_state": "state_name or null"
        }}
        """

        try:
            if self.provider == LLMProvider.ANTHROPIC:
                response = await self._anthropic_completion(prompt, context)
            else:
                response = await self._openai_completion(prompt, context)

            return Intent(**json.loads(response))
        except Exception as e:
            raise ValueError(f"Error extracting intent: {str(e)}")

    async def generate_response(self, message: str, intent: Intent, context: str = "") -> str:
        """Generate natural language response based on intent"""
        prompt = f"""
        Generate a natural response to the user's message.

        Message: "{message}"
        Extracted Intent: {intent.json()}

        Requirements:
        1. Be professional and helpful
        2. Address all identified entities
        3. Ask for clarification if needed
        4. Provide next steps or instructions
        5. Keep the conversation flowing naturally
        """

        if self.provider == LLMProvider.ANTHROPIC:
            return await self._anthropic_completion(prompt, context)
        return await self._openai_completion(prompt, context)

    async def _anthropic_completion(self, prompt: str, context: str = "") -> str:
        """Get completion from Anthropic's Claude"""
        messages = [{
            "role": "system",
            "content": self._get_system_prompt(context)
        }, {
            "role": "user",
            "content": prompt
        }]

        response = await self.client.messages.create(
            model=self.model_config.get("model", self.default_model),
            max_tokens=self.model_config.get("max_tokens", 1000),
            temperature=self.model_config.get("temperature", 0.7),
            messages=messages
        )
        return response.content[0].text

    async def _openai_completion(self, prompt: str, context: str = "") -> str:
        """Get completion from OpenAI's GPT"""
        response = self.client.chat.completions.create(
            model=self.model_config.get("model", self.default_model),
            messages=[
                {"role": "system", "content": self._get_system_prompt(context)},
                {"role": "user", "content": prompt}
            ],
            max_tokens=self.model_config.get("max_tokens", 1000),
            temperature=self.model_config.get("temperature", 0.7)
        )
        return response.choices[0].message.content

    async def handle_special_cases(self, message: str, context: str = "") -> Dict:
        """Handle special cases and complex queries"""
        prompt = f"""
        Analyze this message for special handling requirements:
        "{message}"

        Identify if this requires:
        1. Multiple product handling
        2. Complex shipping requirements
        3. Special pricing or discounts
        4. Order modifications
        5. Custom requests

        Return a JSON object with handling instructions.
        """

        response = await self.generate_response(prompt, None, context)
        return json.loads(response)

    def add_to_history(self, message: Message):
        """Add message to conversation history"""
        self.conversation_history.append(message)

    def get_conversation_summary(self) -> str:
        """Generate a summary of the conversation"""
        messages = [msg.content for msg in self.conversation_history[-5:]]
        return "\n".join(messages)

    async def clarify_ambiguity(self, message: str, ambiguous_entities: List[str]) -> str:
        """Generate a clarifying question for ambiguous input"""
        prompt = f"""
        The following entities are ambiguous in the message:
        {', '.join(ambiguous_entities)}

        Message: "{message}"

        Generate a natural clarifying question to resolve the ambiguity.
        """

        return await self.generate_response(prompt, None)

    @lru_cache(maxsize=100)
    def get_prompt_template(self, template_name: str) -> str:
        """Get cached prompt template"""
        templates = {
            "order_confirmation": """
                Please confirm the following order details:
                - Customer: {customer_name}
                - Products: {products}
                - Shipping to: {shipping_address}
                - Total Amount: ${total_amount}

                Is this correct? Would you like to proceed with the order?
            """,
            "error_handling": """
                I apologize, but I encountered an error: {error_message}

                Would you like to:
                1. Try again
                2. Start over
                3. Speak with a human representative
            """,
            # Add more templates as needed
        }
        return templates.get(template_name, "")

    async def validate_entities(self, entities: EntityExtraction) -> Dict[str, bool]:
        """Validate extracted entities"""
        validations = {
            "customer": bool(entities.customer_name or entities.customer_id),
            "products": len(entities.products) > 0 and all(
                "name" in p and "quantity" in p for p in entities.products
            ),
            "shipping": bool(entities.shipping_address)
        }
        return validations

    async def handle_error(self, error: Exception, context: str = "") -> str:
        """Generate appropriate error response"""
        prompt = f"""
        An error occurred: {str(error)}

        Generate a helpful error message that:
        1. Apologizes for the issue
        2. Explains what went wrong
        3. Suggests next steps
        4. Maintains a professional tone
        """

        return await self.generate_response(prompt, None, context)