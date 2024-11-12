from typing import Dict, Optional, List
from enum import Enum
from datetime import datetime
from anthropic import Anthropic
import openai
from .database import DatabaseManager
from .models import OrderContext, OrderStatus
from .bulk_order import BulkOrderManager
from typing import Tuple
from .utils import logger, safe_json_loads, parse_openai_content


class ConversationState(Enum):
    # Initial states
    INIT = "INIT"
    GREETING = "GREETING"

    # Customer identification states
    CUSTOMER_SELECTION = "CUSTOMER_SELECTION"
    CUSTOMER_CONFIRMATION = "CUSTOMER_CONFIRMATION"

    # Order creation states
    ORDER_START = "ORDER_START"
    PRODUCT_SELECTION = "PRODUCT_SELECTION"
    QUANTITY_INPUT = "QUANTITY_INPUT"
    SHIPPING_ADDRESS = "SHIPPING_ADDRESS"

    # Additional product states
    ADD_MORE_PRODUCTS = "ADD_MORE_PRODUCTS"

    # Order finalization states
    ORDER_SUMMARY = "ORDER_SUMMARY"
    PRICE_CONFIRMATION = "PRICE_CONFIRMATION"
    PAYMENT_METHOD = "PAYMENT_METHOD"
    ORDER_CONFIRMATION = "ORDER_CONFIRMATION"

    # Special instruction states
    SPECIAL_INSTRUCTIONS = "SPECIAL_INSTRUCTIONS"

    # Completion states
    ORDER_PROCESSING = "ORDER_PROCESSING"
    ORDER_COMPLETE = "ORDER_COMPLETE"

    # Error and utility states
    ERROR = "ERROR"
    CLARIFICATION = "CLARIFICATION"
    CANCEL = "CANCEL"


class LLMEnhancedChatbot:
    def __init__(self, database_url: str, llm_provider="anthropic"):
        self.db_manager = DatabaseManager(database_url)
        self.bulk_order_manager = BulkOrderManager(self.db_manager)
        self.state = ConversationState.INIT
        self.order_context = OrderContext()
        self.conversation_history = []
        self.llm_provider = llm_provider

        if llm_provider == "anthropic":
            self.client = Anthropic()
        else:
            self.client = openai.OpenAI()

    def create_system_prompt(self) -> str:
        return f"""Eres un asistente de ventas AI que ayuda a crear pedidos.
        Estado Actual: {self.state.value}
        Contexto del Pedido: {self.order_context.json() if self.order_context else "Sin pedido activo"}

        Basado en el estado actual, debes:

        {self._get_state_instructions()}

        Mantén siempre un tono profesional y servicial.
        Si necesitas aclaraciones, haz preguntas específicas.
        Guía al usuario a través del proceso de pedido paso a paso.
        """
    def _get_state_instructions(self) -> str:
        """Obtener instrucciones específicas basadas en el estado actual"""
        instructions = {
            ConversationState.INIT: """
                - Saluda al usuario
                - Pregunta si están listos para hacer un pedido
                - Cambia al estado GREETING
            """,
            ConversationState.GREETING: """
                - Identifica si el usuario es cliente recurrente
                - Solicita información del cliente
                - Cambia al estado CUSTOMER_SELECTION
            """,
            ConversationState.CUSTOMER_SELECTION: """
                - Valida la información del cliente
                - Si es válida, cambia a CUSTOMER_CONFIRMATION
                - Si no es válida, solicita aclaración
            """,
            ConversationState.CUSTOMER_CONFIRMATION: """
                - Confirma los detalles del cliente
                - Cambia al estado ORDER_START
            """,
            ConversationState.ORDER_START: """
                - Pregunta qué productos desean pedir
                - Cambia al estado PRODUCT_SELECTION
            """,
            ConversationState.PRODUCT_SELECTION: """
                - Ayuda a seleccionar productos específicos
                - Valida disponibilidad del producto
                - Cambia al estado QUANTITY_INPUT
            """,
            ConversationState.QUANTITY_INPUT: """
                - Obtén la cantidad deseada
                - Valida disponibilidad de inventario
                - Pregunta si quieren agregar más productos
            """,
            ConversationState.ADD_MORE_PRODUCTS: """
                - Si es sí, vuelve a PRODUCT_SELECTION
                - Si es no, cambia a SHIPPING_ADDRESS
            """,
            ConversationState.SHIPPING_ADDRESS: """
                - Confirma o actualiza dirección de envío
                - Cambia al estado SPECIAL_INSTRUCTIONS
            """,
            ConversationState.SPECIAL_INSTRUCTIONS: """
                - Pregunta por instrucciones especiales
                - Cambia al estado ORDER_SUMMARY
            """,
            ConversationState.ORDER_SUMMARY: """
                - Muestra resumen completo del pedido
                - Cambia al estado PRICE_CONFIRMATION
            """,
            ConversationState.PRICE_CONFIRMATION: """
                - Confirma detalles de precios
                - Cambia al estado PAYMENT_METHOD
            """,
            ConversationState.PAYMENT_METHOD: """
                - Obtén método de pago
                - Cambia al estado ORDER_CONFIRMATION
            """,
            ConversationState.ORDER_CONFIRMATION: """
                - Obtén confirmación final
                - Si se confirma, cambia a ORDER_PROCESSING
                - Si no, permite modificaciones
            """,
            ConversationState.ORDER_PROCESSING: """
                - Procesa el pedido
                - Actualiza inventario
                - Cambia al estado ORDER_COMPLETE
            """,
            ConversationState.ORDER_COMPLETE: """
                - Muestra confirmación del pedido
                - Proporciona número de pedido
                - Pregunta si necesitan algo más
            """
        }
        return instructions.get(self.state, "Continúa con la conversación de manera natural.")

    async def extract_intent_and_entities(self, message: str) -> Dict:
        prompt = f"""
        Analiza el siguiente mensaje y extrae información relevante para el procesamiento del pedido.
        Estado actual: {self.state.value}
        Mensaje: "{message}"

        Devuelve un objeto JSON con:
        - intent: la intención principal del usuario
        - entities: entidades relevantes (clientes, productos, cantidades, direcciones)
        - requires_clarification: booleano
        - suggested_next_state: siguiente estado de la conversación
        """

        if self.llm_provider == "anthropic":
            response = await self.client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1000,
                temperature=0,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            return safe_json_loads(response.content[0].text)
        else:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": self.create_system_prompt()},
                    {"role": "user", "content": prompt}
                ]
            )
            content_response = parse_openai_content(response.choices[0].message.content)
            print ("CONTENT_PARSED", content_response)
            return safe_json_loads(content_response)

    async def _identify_customer(self, extracted_info: Dict) -> Optional[Dict]:
        """Identify customer from extracted information"""
        if customer_id := extracted_info.get('entities', {}).get('customer_id'):
            return await self.db_manager.get_customer(customer_id)
        return None

    async def _process_customer_info(self, extracted_info: Dict) -> Optional[Dict]:
        """Process customer information"""
        # Try to find existing customer or create new one
        customer = False

        if customer_id := extracted_info.get('entities', {}).get('customer_id'):
            customer = await self.db_manager.get_customer(customer_id)

        if customer_email := extracted_info.get('entities', {}).get('email'):
            customer = await self.db_manager.get_customer_by_email(customer_email)

        if not customer:
            customer = await self.db_manager.create_customer(extracted_info)

        self.order_context.customer_id = customer['id']
        return customer

    async def _process_product_selection(self, extracted_info: Dict) -> Optional[Dict]:
        """Process product selection"""
        product_name = extracted_info.get('entities', {}).get('product_name')
        if product_name:
            product = await self.db_manager.get_product_by_name(product_name)

        product = extracted_info.get('entities', {}).get('product')
        if product:
            product = await self.db_manager.get_product_by_name(product)

        product_id = extracted_info.get('entities', {}).get('product_id')
        if product_id:
            product = await self.db_manager.get_product(product_id)

        print("PRODUCT ", product)
        if product:
            if not self.order_context.items:
                self.order_context.items = []

            self.order_context.items.append({
                'product_id': product['id'],
                'product_name': product['name'],
                'unit_price': product['price'],
                'quantity': 0  # Will be updated in quantity input state
            })
        else:
            return None

        return product

    async def _validate_quantity(self, extracted_info: Dict) -> bool:
        """Validate requested quantity against inventory"""
        quantity = extracted_info.get('entities', {}).get('quantity')
        if not quantity or not self.order_context.items:
            return False

        current_item = self.order_context.items[-1]
        is_valid, _ = await self.db_manager.validate_inventory(
            current_item['product_id'],
            quantity
        )

        if is_valid:
            current_item['quantity'] = quantity
            self._update_total_amount()
            return True

        return False

    def _wants_more_products(self, message: str) -> bool:
        """Check if user wants to add more products"""
        positive_responses = {'yes', 'yeah', 'sure', 'ok', 'okay', 'yep', 'y'}
        return message.lower().strip() in positive_responses

    async def _process_shipping_address(self, extracted_info: Dict) -> bool:
        """Process shipping address"""
        address = extracted_info.get('entities', {}).get('shipping_address')
        if not address and self.order_context.customer_id:
            customer = await self.db_manager.get_customer(self.order_context.customer_id)
            address = customer.get('default_shipping_address')

        if address:
            self.order_context.shipping_address = address
            return True
        return False

    def _add_special_instructions(self, message: str):
        """Add special instructions to order"""
        if message.lower().strip() not in {'no', 'none', 'n'}:
            self.order_context.special_instructions = message

    async def _generate_order_summary(self) -> str:
        """Generar resumen del pedido"""
        summary = "Resumen del Pedido:\n"

        if self.order_context.customer_id:
            customer = await self.db_manager.get_customer(self.order_context.customer_id)
            summary += f"Cliente: {customer['name']}\n"

        summary += f"Envío a: {self.order_context.shipping_address}\n\n"

        summary += "Productos:\n"
        for item in self.order_context.items:
            subtotal = item['quantity'] * item['unit_price']
            summary += f"- {item['quantity']}x {item['product_name']} @ ${item['unit_price']} = ${subtotal}\n"

        summary += f"\nMonto Total: ${self.order_context.total_amount:.2f}"

        if self.order_context.special_instructions:
            summary += f"\n\nInstrucciones Especiales: {self.order_context.special_instructions}"

        return summary

    async def generate_response(self, message: str, extracted_info: Dict) -> str:
        prompt = f"""
        Genera una respuesta natural al mensaje del usuario.
        Estado actual: {self.state.value}
        Mensaje del usuario: "{message}"
        Información extraída: {json.dumps(extracted_info)}

        Requisitos:
        1. Sé conversacional pero profesional
        2. Si falta información, solicítala
        3. Si se procede al siguiente estado, proporciona instrucciones claras
        4. Si hay problemas, explícalos claramente
        """

        if self.llm_provider == "anthropic":
            response = await self.client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1000,
                temperature=0.7,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            return response.content[0].text
        else:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": self.create_system_prompt()},
                    {"role": "user", "content": prompt}
                ]
            )
            content_response = parse_openai_content(response.choices[0].message.content)
            print("CONTENT_PARSED", content_response)
            return safe_json_loads(content_response)

    def _confirms_price(self, message: str) -> bool:
        """Verificar si el usuario confirma el precio"""
        positive_responses = {'si', 'sí', 'claro', 'ok', 'okay', 'correcto', 's', 'proceder'}
        return message.lower().strip() in positive_responses

    def _confirms_order(self, message: str) -> bool:
        """Verificar si el usuario confirma el pedido"""
        positive_responses = {'si', 'sí', 'claro', 'ok', 'okay', 'confirmar', 's', 'realizar pedido'}
        return message.lower().strip() in positive_responses

    async def _process_order(self) -> str:
        """Procesar el pedido antes de la confirmación final"""
        try:
            # Validar todos los artículos una última vez
            invalid_items = []
            for item in self.order_context.items:
                is_valid, message = await self.db_manager.validate_inventory(
                    item['product_id'],
                    item['quantity']
                )
                if not is_valid:
                    invalid_items.append(f"{item['product_name']}: {message}")

            if invalid_items:
                self.state = ConversationState.ERROR
                return "Algunos artículos ya no están disponibles:\n" + "\n".join(invalid_items)

            # Validar dirección de envío
            if not self.order_context.shipping_address:
                self.state = ConversationState.ERROR
                return "Falta la dirección de envío. Por favor proporcione una dirección válida."

            # Calcular totales finales
            self._update_total_amount()

            return f"Pedido validado. Monto total: ${self.order_context.total_amount:.2f}\n¿Desea proceder con la creación del pedido?"

        except Exception as e:
            logger.error(f"Error procesando el pedido: {str(e)}")
            self.state = ConversationState.ERROR
            return f"Error al procesar el pedido: {str(e)}"

    def _update_total_amount(self):
        """Update the total amount of the order"""
        total = sum(
            item['quantity'] * item['unit_price']
            for item in self.order_context.items
        )
        self.order_context.total_amount = total

    async def _finalize_order(self) -> Dict:
        """Create the final order in the database"""
        try:
            # Create order items format for database
            items = [
                {
                    "product_id": item["product_id"],
                    "quantity": item["quantity"]
                }
                for item in self.order_context.items
            ]

            # Create order in database
            order = await self.db_manager.create_order(
                customer_id=self.order_context.customer_id,
                items=items,
                shipping_address=self.order_context.shipping_address,
                special_instructions=self.order_context.special_instructions
            )

            # Reset order context
            self._reset_order_context()

            return order

        except Exception as e:
            logger.error(f"Error finalizing order: {str(e)}")
            self.state = ConversationState.ERROR
            raise

    def _reset_order_context(self):
        """Reset the order context after completion"""
        self.order_context = OrderContext()

    async def _handle_clarification(self, message: str) -> str:
        """Manejar casos donde se necesita aclaración"""
        previous_state = self.state
        self.state = ConversationState.CLARIFICATION

        clarification_prompts = {
            ConversationState.CUSTOMER_SELECTION:
                "¿Podría proporcionarme su nombre y dirección de correo electrónico?",
            ConversationState.PRODUCT_SELECTION:
                "¿Podría especificar qué producto le interesa?",
            ConversationState.QUANTITY_INPUT:
                "Por favor, especifique cuántas unidades desea ordenar.",
            ConversationState.SHIPPING_ADDRESS:
                "Por favor, proporcione una dirección de envío completa.",
        }

        response = clarification_prompts.get(
            previous_state,
            "¿Podría aclarar su solicitud?"
        )

        self.state = previous_state
        return response

    async def _handle_modification(self, message: str, extracted_info: Dict) -> str:
        """Manejar modificaciones del pedido"""
        modification_type = extracted_info.get('entities', {}).get('modification_type')

        if modification_type == 'quantity':
            return await self._modify_quantity(extracted_info)
        elif modification_type == 'product':
            return await self._modify_product(extracted_info)
        elif modification_type == 'shipping':
            return await self._modify_shipping(extracted_info)
        else:
            return "¿Qué desea modificar? (cantidad, producto o dirección de envío)"

    async def _modify_quantity(self, extracted_info: Dict) -> str:
        """Modificar cantidad de un artículo existente"""
        product_id = extracted_info.get('entities', {}).get('product_id')
        new_quantity = extracted_info.get('entities', {}).get('quantity')

        if not product_id or not new_quantity:
            return "Por favor, especifique qué producto y la nueva cantidad."

        for item in self.order_context.items:
            if item['product_id'] == product_id:
                is_valid, message = await self.db_manager.validate_inventory(
                    product_id,
                    new_quantity
                )
                if is_valid:
                    item['quantity'] = new_quantity
                    self._update_total_amount()
                    return f"Cantidad actualizada. Nuevo total: ${self.order_context.total_amount:.2f}"
                return f"No se puede actualizar la cantidad: {message}"

        return "Producto no encontrado en su pedido."

    async def _modify_product(self, extracted_info: Dict) -> str:
        """Modificar o eliminar un producto"""
        product_id = extracted_info.get('entities', {}).get('product_id')
        action = extracted_info.get('entities', {}).get('action', 'remove')

        if action == 'remove':
            self.order_context.items = [
                item for item in self.order_context.items
                if item['product_id'] != product_id
            ]
            self._update_total_amount()
            return f"Producto eliminado. Nuevo total: ${self.order_context.total_amount:.2f}"

        return "Por favor, especifique qué desea hacer con el producto."

    async def _modify_shipping(self, extracted_info: Dict) -> str:
        """Modificar dirección de envío"""
        new_address = extracted_info.get('entities', {}).get('shipping_address')

        if new_address:
            self.order_context.shipping_address = new_address
            return "Dirección de envío actualizada."

        return "Por favor proporcione la nueva dirección de envío."

    def _get_current_state_message(self) -> str:
        """Obtener mensaje apropiado para el estado actual"""
        state_messages = {
            ConversationState.INIT:
                "¡Bienvenido! ¿Cómo puedo ayudarle hoy?",
            ConversationState.CUSTOMER_SELECTION:
                "Por favor, proporcione su información de cliente.",
            ConversationState.PRODUCT_SELECTION:
                "¿Qué producto desea ordenar?",
            ConversationState.QUANTITY_INPUT:
                "¿Cuántas unidades desea?",
            ConversationState.SHIPPING_ADDRESS:
                "¿A dónde debemos enviar su pedido?",
            ConversationState.ORDER_SUMMARY:
                "Aquí está el resumen de su pedido.",
            ConversationState.ORDER_CONFIRMATION:
                "¿Desea confirmar este pedido?",
            ConversationState.ERROR:
                "Me disculpo, pero hubo un error. ¿Desea intentarlo de nuevo?"
        }
        return state_messages.get(self.state, "¿Cómo puedo ayudarle?")

    def _log_state_transition(self, previous_state: ConversationState):
        """Log state transitions for debugging"""
        logger.info(
            f"State transition: {previous_state.value} -> {self.state.value}"
        )

    async def _save_conversation_state(self) -> Dict:
        """Save current conversation state"""
        return {
            "state": self.state.value,
            "order_context": self.order_context.dict(),
            "conversation_history": self.conversation_history,
            "timestamp": datetime.now().isoformat()
        }

    async def _restore_conversation_state(self, state_data: Dict) -> bool:
        """Restore conversation from saved state"""
        try:
            self.state = ConversationState(state_data["state"])
            self.order_context = OrderContext(**state_data["order_context"])
            self.conversation_history = state_data["conversation_history"]
            return True
        except Exception as e:
            logger.error(f"Error restoring conversation state: {str(e)}")
            return False

    async def _handle_state(self, message: str, extracted_info: Dict) -> str:
        """Manejar la conversación basada en el estado actual"""
        print("STATE ", self.state)
        try:
            if self.state == ConversationState.INIT:
                if customer := await self._identify_customer(extracted_info):
                    self.order_context.customer_id = customer.get("id")
                    self.state = ConversationState.CUSTOMER_CONFIRMATION
                    return f"¡Bienvenido de nuevo {customer.get('name')}! ¿Desea realizar un nuevo pedido?"
                else:
                    self.state = ConversationState.GREETING
                    return "¡Bienvenido! Estoy aquí para ayudarle a realizar su pedido. ¿Es usted un cliente recurrente?"

            elif self.state == ConversationState.GREETING:
                # Manejar identificación del cliente
                if customer := await self._identify_customer(extracted_info):
                    self.order_context.customer_id = customer.get("id")
                    self.state = ConversationState.ORDER_START
                    return f"¡Bienvenido de nuevo {customer.get('name')}! ¿Desea realizar un nuevo pedido?"
                else:
                    self.state = ConversationState.CUSTOMER_SELECTION
                    return "¿Podría proporcionarme su información de cliente?"

            elif self.state == ConversationState.CUSTOMER_CONFIRMATION:
                # Manejar confirmación del cliente
                if extracted_info["suggested_next_state"]:
                    new_state = ConversationState(extracted_info["suggested_next_state"])
                    print("NEW_STATE ", new_state)
                    if self.is_valid_state_transition(new_state):
                        self.state = new_state
                        return "¡Excelente! ¿Qué productos desea ordenar?"
                    else:
                        self.state = ConversationState.CUSTOMER_SELECTION
                        return "¿Cómo puedo ayudarle?"
                else:
                    self.state = ConversationState.CUSTOMER_SELECTION
                    return "¿Podría proporcionarme su información de cliente?"

            elif self.state == ConversationState.CUSTOMER_SELECTION:
                # Manejar registro de nuevo cliente o búsqueda
                customer_info = await self._process_customer_info(extracted_info)
                if customer_info:
                    self.state = ConversationState.ORDER_START
                    return "He encontrado su información. ¿Desea proceder con la realización del pedido?"
                return "No pude encontrar su información. ¿Podría proporcionar más detalles?"

            elif self.state == ConversationState.ORDER_START:
                self.state = ConversationState.PRODUCT_SELECTION
                return "¿Qué productos desea ordenar?"

            elif self.state == ConversationState.PRODUCT_SELECTION:
                # Manejar selección de producto
                if product := await self._process_product_selection(extracted_info):
                    self.state = ConversationState.QUANTITY_INPUT
                    return f"¿Cuántas unidades de {product.get('name')} desea ordenar?"
                return "No pude encontrar ese producto. ¿Podría especificar otro producto?"

            elif self.state == ConversationState.QUANTITY_INPUT:
                # Manejar entrada de cantidad y validación
                if await self._validate_quantity(extracted_info):
                    self.state = ConversationState.ADD_MORE_PRODUCTS
                    return "¿Desea agregar más productos a su pedido?"
                return "Lo siento, esa cantidad no está disponible. Por favor, intente con una cantidad diferente."

            elif self.state == ConversationState.ADD_MORE_PRODUCTS:
                if self._wants_more_products(message):
                    self.state = ConversationState.PRODUCT_SELECTION
                    return "¿Qué otros productos desea agregar?"
                self.state = ConversationState.SHIPPING_ADDRESS
                return "¡Perfecto! ¿Debemos usar su dirección de envío predeterminada?"

            elif self.state == ConversationState.SHIPPING_ADDRESS:
                # Manejar confirmación/actualización de dirección de envío
                if await self._process_shipping_address(extracted_info):
                    self.state = ConversationState.SPECIAL_INSTRUCTIONS
                    return "¿Desea agregar alguna instrucción especial para este pedido?"
                return "¿Podría proporcionar una dirección de envío válida?"

            elif self.state == ConversationState.SPECIAL_INSTRUCTIONS:
                self._add_special_instructions(message)
                self.state = ConversationState.ORDER_SUMMARY
                return await self._generate_order_summary()

            elif self.state == ConversationState.ORDER_SUMMARY:
                self.state = ConversationState.PRICE_CONFIRMATION
                return "El total de su pedido será ${:.2f}. ¿Desea proceder?".format(
                    self.order_context.total_amount or 0
                )

            elif self.state == ConversationState.PRICE_CONFIRMATION:
                if self._confirms_price(message):
                    self.state = ConversationState.ORDER_CONFIRMATION
                    return "¿Desea confirmar este pedido?"
                self.state = ConversationState.CANCEL
                return "¿Desea modificar el pedido o cancelarlo?"

            elif self.state == ConversationState.ORDER_CONFIRMATION:
                if self._confirms_order(message):
                    self.state = ConversationState.ORDER_PROCESSING
                    return await self._process_order()
                self.state = ConversationState.CANCEL
                return "Pedido cancelado. ¿Desea comenzar de nuevo?"

            elif self.state == ConversationState.ORDER_PROCESSING:
                order_result = await self._finalize_order()
                self.state = ConversationState.ORDER_COMPLETE
                return f"¡Su pedido ha sido realizado con éxito! ID del Pedido: {order_result['order_id']}"

            elif self.state == ConversationState.ERROR:
                self.state = ConversationState.INIT
                return "Me disculpo por el error. ¿Desea comenzar de nuevo?"

            else:
                return await self.generate_response(message, extracted_info)

        except Exception as e:
            logger.error(f"Error handling state {self.state}: {str(e)}")
            self.state = ConversationState.ERROR
            return f"Me disculpo, pero algo salió mal. ¿Desea intentarlo de nuevo?"

    async def process_message(self, message: str) -> str:
        try:
            # Add message to conversation history
            self.conversation_history.append({
                "role": "user",
                "content": message,
                "timestamp": datetime.now()
            })

            # Extract intent and entities
            extracted_info = await self.extract_intent_and_entities(message)
            print("EXTRACTED_INFO", extracted_info)

            print("MESSAGE", message)
            # Process based on current state
            response = await self._handle_state(message, extracted_info)
            print("RESPONSE", response)

            # Add response to conversation history
            self.conversation_history.append({
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now()
            })

            return response

        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            self.state = ConversationState.ERROR
            return f"Disculpe, encontre un error: {str(e)}. ¿Desea intentarlo de nuevo?"

    def is_valid_state_transition(self, new_state: ConversationState) -> bool:
        valid_transitions = {
            ConversationState.INIT: [ConversationState.INIT],
            ConversationState.GREETING: [ConversationState.GREETING],
            ConversationState.CUSTOMER_SELECTION: [ConversationState.CUSTOMER_SELECTION],
            ConversationState.CUSTOMER_CONFIRMATION: [ConversationState.CUSTOMER_CONFIRMATION],
            ConversationState.ORDER_START: [ConversationState.ORDER_START],
            ConversationState.PRODUCT_SELECTION: [ConversationState.PRODUCT_SELECTION],
            ConversationState.QUANTITY_INPUT: [ConversationState.QUANTITY_INPUT],
            ConversationState.SHIPPING_ADDRESS: [ConversationState.SHIPPING_ADDRESS],
            ConversationState.ADD_MORE_PRODUCTS: [ConversationState.ADD_MORE_PRODUCTS],
            ConversationState.ORDER_SUMMARY: [ConversationState.ORDER_SUMMARY],
            ConversationState.PRICE_CONFIRMATION: [ConversationState.PRICE_CONFIRMATION],
            ConversationState.PAYMENT_METHOD: [ConversationState.PAYMENT_METHOD],
            ConversationState.ORDER_CONFIRMATION: [ConversationState.ORDER_CONFIRMATION],
            ConversationState.SPECIAL_INSTRUCTIONS: [ConversationState.SPECIAL_INSTRUCTIONS],
            ConversationState.ORDER_PROCESSING: [ConversationState.ORDER_PROCESSING],
            ConversationState.ORDER_COMPLETE: [ConversationState.ORDER_COMPLETE],
            ConversationState.ERROR: [ConversationState.ERROR],
            ConversationState.CLARIFICATION: [ConversationState.CLARIFICATION],
            ConversationState.CANCEL: [ConversationState.CANCEL]
        }

        return new_state in valid_transitions.get(self.state, [])

    def get_conversation_history(self) -> List[Dict]:
        return self.conversation_history

    async def handle_customer_selection(self, customer_id: int) -> Optional[Dict]:
        """Handle customer selection and validation"""
        customer = await self.db_manager.get_customer(customer_id)
        if customer:
            self.current_customer = customer
            self.state = ConversationState.PRODUCT_SELECTION
            return customer
        return None

    async def handle_product_selection(self, product_id: int, quantity: int) -> Tuple[bool, str]:
        """Handle product selection and inventory validation"""
        product = await self.db_manager.get_product(product_id)
        if not product:
            return False, "Product not found"

        # Validate inventory
        is_valid, message = await self.db_manager.validate_inventory(product_id, quantity)
        if not is_valid:
            return False, message

        self.current_product = product
        self.current_quantity = quantity
        self.state = ConversationState.SHIPPING_ADDRESS
        return True, "Product available"

    async def handle_shipping_address(self, address: Optional[str] = None) -> str:
        """Handle shipping address confirmation or update"""
        if not hasattr(self, 'current_customer'):
            return "Please select a customer first"

        if address:
            self.shipping_address = address
        else:
            self.shipping_address = self.current_customer.get('default_shipping_address')

        self.state = ConversationState.CONFIRMATION
        return self.shipping_address

    async def handle_order_confirmation(self) -> Dict:
        """Process final order confirmation"""
        try:
            order_data = {
                "customer_id": self.current_customer['id'],
                "items": [{
                    "product_id": self.current_product['id'],
                    "quantity": self.current_quantity
                }],
                "shipping_address": self.shipping_address
            }

            order = await self.db_manager.create_order(**order_data)
            self.state = ConversationState.COMPLETE
            self.reset_order_context()
            return order

        except Exception as e:
            self.state = ConversationState.ERROR
            raise ValueError(f"Order confirmation failed: {str(e)}")

    def reset_order_context(self):
        """Reset the current order context"""
        self.current_customer = None
        self.current_product = None
        self.current_quantity = None
        self.shipping_address = None
        self.state = ConversationState.INIT

    async def format_order_summary(self) -> str:
        """Generar un resumen del pedido formateado"""
        if not all([hasattr(self, attr) for attr in
                    ['current_customer', 'current_product', 'current_quantity', 'shipping_address']]):
            return "No se puede generar el resumen del pedido - falta información"

        total_amount = self.current_product['price'] * self.current_quantity

        summary = "Resumen del Pedido:\n"
        summary += f"Cliente: {self.current_customer['name']}\n"
        summary += f"Producto: {self.current_product['name']}\n"
        summary += f"Cantidad: {self.current_quantity}\n"
        summary += f"Precio Unitario: ${self.current_product['price']:.2f}\n"
        summary += f"Monto Total: ${total_amount:.2f}\n"
        summary += f"Dirección de Envío: {self.shipping_address}\n"

        return summary

    async def handle_special_requests(self, message: str) -> str:
        """Manejar solicitudes o preguntas especiales"""
        prompt = f"""
        El usuario tiene una solicitud o pregunta especial:
        "{message}"

        Genera una respuesta útil que:
        1. Aborde su solicitud específica
        2. Mantenga el contexto actual del pedido
        3. Los guíe de vuelta al proceso de pedido si es necesario
        """

        if self.llm_provider == "anthropic":
            response = await self.client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1000,
                temperature=0.7,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            return response.content[0].text
        else:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": self.create_system_prompt()},
                    {"role": "user", "content": prompt}
                ]
            )
            content_response = parse_openai_content(response.choices[0].message.content)
            print("CONTENT_PARSED", content_response)
            return safe_json_loads(content_response)

    async def handle_error_recovery(self, error_message: str) -> str:
        """Manejar la recuperación de errores y proporcionar orientación"""
        self.state = ConversationState.ERROR

        recovery_prompt = f"""
        Ha ocurrido un error: "{error_message}"

        Genera una respuesta útil que:
        1. Se disculpe por el error
        2. Explique qué salió mal en términos simples
        3. Proporcione pasos claros a seguir
        4. Ofrezca comenzar de nuevo si es necesario
        """

        if self.llm_provider == "anthropic":
            response = await self.client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1000,
                temperature=0.7,
                messages=[{
                    "role": "user",
                    "content": recovery_prompt
                }]
            )
            return response.content[0].text
        else:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": self.create_system_prompt()},
                    {"role": "user", "content": recovery_prompt}
                ]
            )
            content_response = parse_openai_content(response.choices[0].message.content)
            print("CONTENT_PARSED", content_response)
            return safe_json_loads(content_response)

    async def save_conversation_context(self) -> Dict:
        """Save current conversation context for later recovery"""
        context = {
            "state": self.state.value,
            "customer": getattr(self, 'current_customer', None),
            "product": getattr(self, 'current_product', None),
            "quantity": getattr(self, 'current_quantity', None),
            "shipping_address": getattr(self, 'shipping_address', None),
            "conversation_history": self.conversation_history,
            "timestamp": datetime.now().isoformat()
        }
        return context

    async def restore_conversation_context(self, context: Dict) -> bool:
        """Restore conversation from saved context"""
        try:
            self.state = ConversationState(context['state'])
            if context.get('customer'):
                self.current_customer = context['customer']
            if context.get('product'):
                self.current_product = context['product']
            if context.get('quantity'):
                self.current_quantity = context['quantity']
            if context.get('shipping_address'):
                self.shipping_address = context['shipping_address']
            self.conversation_history = context.get('conversation_history', [])
            return True
        except Exception as e:
            print(f"Error restoring context: {str(e)}")
            self.reset_order_context()
            return False

    async def get_order_status(self, order_id: int) -> Optional[Dict]:
        """Get current status of an order"""
        with self.db_manager.get_db() as db:
            order = db.query(Order).filter(Order.id == order_id).first()
            if order:
                return {
                    "order_id": order.id,
                    "status": order.status.value,
                    "total_amount": order.total_amount,
                    "created_at": order.created_at,
                    "updated_at": order.updated_at
                }
            return None

    async def handle_order_modification(self, order_id: int, modifications: Dict) -> Tuple[bool, str]:
        """Handle modifications to existing orders"""
        # This would need to be implemented based on your specific business rules
        # for order modifications
        pass