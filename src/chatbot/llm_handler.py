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
        """Obtener configuración predeterminada para el LLM"""
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
        """Obtener prompt del sistema en caché con contexto"""
        return f"""Eres un asistente de ventas AI que ayuda con el procesamiento de pedidos.
        Tu función es comprender las solicitudes de los clientes y extraer información relevante.

        Contexto Actual:
        {context}

        Directrices:
        1. Extraer detalles del cliente, productos, cantidades e información de envío
        2. Identificar la intención principal de cada mensaje
        3. Mantener un tono profesional y servicial
        4. Solicitar aclaraciones cuando sea necesario
        5. Manejar pedidos complejos y solicitudes especiales apropiadamente

        Acciones Disponibles:
        - Procesamiento de pedidos
        - Consultas de productos
        - Información de envío
        - Modificaciones de pedidos
        - Verificaciones de estado
        """

    async def extract_intent(self, message: str, context: str = "") -> Intent:
        """Extraer intención y entidades del mensaje del usuario"""
        prompt = f"""
        Analiza el siguiente mensaje y extrae información relacionada con el pedido.

        Mensaje: "{message}"

        Devuelve un objeto JSON con:
        {{
            "primary": "tipo_de_intencion",
            "confidence": 0.0 a 1.0,
            "entities": {{
                "customer_name": "nombre o null",
                "customer_id": número o null,
                "products": [
                    {{"name": "nombre_producto", "quantity": número}}
                ],
                "shipping_address": "dirección o null",
                "special_instructions": "instrucciones o null"
            }},
            "requires_clarification": booleano,
            "suggested_next_state": "nombre_estado o null"
        }}
        """

        try:
            if self.provider == LLMProvider.ANTHROPIC:
                response = await self._anthropic_completion(prompt, context)
            else:
                response = await self._openai_completion(prompt, context)

            return Intent(**json.loads(response))
        except Exception as e:
            raise ValueError(f"Error al extraer la intención: {str(e)}")

    async def generate_response(self, message: str, intent: Intent, context: str = "") -> str:
        """Generar respuesta en lenguaje natural basada en la intención"""
        prompt = f"""
        Genera una respuesta natural al mensaje del usuario.

        Mensaje: "{message}"
        Intención Extraída: {intent.json()}

        Requisitos:
        1. Ser profesional y servicial
        2. Abordar todas las entidades identificadas
        3. Solicitar aclaraciones si es necesario
        4. Proporcionar pasos siguientes o instrucciones
        5. Mantener un flujo natural de conversación
        """

        if self.provider == LLMProvider.ANTHROPIC:
            return await self._anthropic_completion(prompt, context)
        return await self._openai_completion(prompt, context)

    async def _anthropic_completion(self, prompt: str, context: str = "") -> str:
        """Obtener respuesta de Claude de Anthropic"""
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
        """Obtener respuesta de GPT de OpenAI"""
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
        """Manejar casos especiales y consultas complejas"""
        prompt = f"""
        Analiza este mensaje para requisitos de manejo especial:
        "{message}"

        Identifica si esto requiere:
        1. Manejo de múltiples productos
        2. Requisitos complejos de envío
        3. Precios especiales o descuentos
        4. Modificaciones de pedidos
        5. Solicitudes personalizadas

        Devuelve un objeto JSON con instrucciones de manejo.
        """

        response = await self.generate_response(prompt, None, context)
        return json.loads(response)

    def add_to_history(self, message: Message):
        """Agregar mensaje al historial de conversación"""
        self.conversation_history.append(message)

    def get_conversation_summary(self) -> str:
        """Generar un resumen de la conversación"""
        messages = [msg.content for msg in self.conversation_history[-5:]]
        return "\n".join(messages)

    async def clarify_ambiguity(self, message: str, ambiguous_entities: List[str]) -> str:
        """Generar una pregunta aclaratoria para entrada ambigua"""
        prompt = f"""
        Las siguientes entidades son ambiguas en el mensaje:
        {', '.join(ambiguous_entities)}

        Mensaje: "{message}"

        Genera una pregunta natural aclaratoria para resolver la ambigüedad.
        """

        return await self.generate_response(prompt, None)

    @lru_cache(maxsize=100)
    def get_prompt_template(self, template_name: str) -> str:
        """Obtener plantilla de prompt en caché"""
        templates = {
            "order_confirmation": """
                Por favor confirma los siguientes detalles del pedido:
                - Cliente: {customer_name}
                - Productos: {products}
                - Envío a: {shipping_address}
                - Monto Total: ${total_amount}

                ¿Es esto correcto? ¿Deseas proceder con el pedido?
            """,
            "error_handling": """
                Me disculpo, pero encontré un error: {error_message}

                ¿Te gustaría:
                1. Intentar de nuevo
                2. Empezar de nuevo
                3. Hablar con un representante humano
            """,
            # Agregar más plantillas según sea necesario
        }
        return templates.get(template_name, "")

    async def validate_entities(self, entities: EntityExtraction) -> Dict[str, bool]:
        """Validar entidades extraídas"""
        validations = {
            "customer": bool(entities.customer_name or entities.customer_id),
            "products": len(entities.products) > 0 and all(
                "name" in p and "quantity" in p for p in entities.products
            ),
            "shipping": bool(entities.shipping_address)
        }
        return validations

    async def handle_error(self, error: Exception, context: str = "") -> str:
        """Generar respuesta apropiada de error"""
        prompt = f"""
        Ocurrió un error: {str(error)}

        Genera un mensaje de error útil que:
        1. Se disculpe por el problema
        2. Explique qué salió mal
        3. Sugiera los siguientes pasos
        4. Mantenga un tono profesional
        """

        return await self.generate_response(prompt, None, context)