from typing import Dict, Optional, List
from enum import Enum
import json
from datetime import datetime
from anthropic import Anthropic
import openai
from .database import DatabaseManager
from .models import OrderStatus
from .bulk_order import BulkOrderManager
from typing import Tuple


class ConversationState(Enum):
    INIT = "init"
    CUSTOMER_SELECTION = "customer_selection"
    PRODUCT_SELECTION = "product_selection"
    QUANTITY_INPUT = "quantity_input"
    SHIPPING_ADDRESS = "shipping_address"
    CONFIRMATION = "confirmation"
    COMPLETE = "complete"
    ERROR = "error"


class LLMEnhancedChatbot:
    def __init__(self, database_url: str, llm_provider="anthropic"):
        self.db_manager = DatabaseManager(database_url)
        self.bulk_order_manager = BulkOrderManager(self.db_manager)
        self.state = ConversationState.INIT
        self.conversation_history = []
        self.llm_provider = llm_provider

        if llm_provider == "anthropic":
            self.client = Anthropic()
        else:
            self.client = openai.OpenAI()

    def create_system_prompt(self) -> str:
        return """You are a helpful sales assistant bot that helps create purchase orders.
        Guide users through the order process naturally while extracting necessary information.

        Required information for order:
        - Customer selection
        - Product selection
        - Quantity
        - Shipping address
        - Final confirmation

        Current conversation state: {state}
        """

    async def extract_intent_and_entities(self, message: str) -> Dict:
        prompt = f"""
        Analyze the following message and extract relevant information for order processing.
        Current state: {self.state.value}
        Message: "{message}"

        Return a JSON object with:
        - intent: the user's primary intent
        - entities: any relevant entities (customers, products, quantities, addresses)
        - requires_clarification: boolean
        - suggested_next_state: next conversation state
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
            return json.loads(response.content[0].text)
        else:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": self.create_system_prompt()},
                    {"role": "user", "content": prompt}
                ]
            )
            print ("RESPONSE", response)
            return json.loads(response.choices[0].message.content)

    async def generate_response(self, message: str, extracted_info: Dict) -> str:
        prompt = f"""
        Generate a natural response to the user's message.
        Current state: {self.state.value}
        User message: "{message}"
        Extracted information: {json.dumps(extracted_info)}

        Requirements:
        1. Be conversational but professional
        2. If information is missing, ask for it
        3. If proceeding to next state, provide clear instructions
        4. If there are issues, explain them clearly
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
            return response.choices[0].message.content

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

            # Handle state transitions
            if extracted_info["suggested_next_state"]:
                new_state = ConversationState(extracted_info["suggested_next_state"])
                if self.is_valid_state_transition(new_state):
                    self.state = new_state

            # Generate and store response
            response = await self.generate_response(message, extracted_info)
            self.conversation_history.append({
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now()
            })

            return response

        except Exception as e:
            self.state = ConversationState.ERROR
            return f"I apologize, but I encountered an error: {str(e)}. Let's start over."

    def is_valid_state_transition(self, new_state: ConversationState) -> bool:
        valid_transitions = {
            ConversationState.INIT: [ConversationState.CUSTOMER_SELECTION],
            ConversationState.CUSTOMER_SELECTION: [ConversationState.PRODUCT_SELECTION],
            ConversationState.PRODUCT_SELECTION: [ConversationState.QUANTITY_INPUT],
            ConversationState.QUANTITY_INPUT: [ConversationState.SHIPPING_ADDRESS],
            ConversationState.SHIPPING_ADDRESS: [ConversationState.CONFIRMATION],
            ConversationState.CONFIRMATION: [ConversationState.COMPLETE, ConversationState.INIT],
            ConversationState.COMPLETE: [ConversationState.INIT],
            ConversationState.ERROR: [ConversationState.INIT]
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
        """Generate a formatted order summary"""
        if not all([hasattr(self, attr) for attr in
                    ['current_customer', 'current_product', 'current_quantity', 'shipping_address']]):
            return "Unable to generate order summary - missing information"

        total_amount = self.current_product['price'] * self.current_quantity

        summary = "Order Summary:\n"
        summary += f"Customer: {self.current_customer['name']}\n"
        summary += f"Product: {self.current_product['name']}\n"
        summary += f"Quantity: {self.current_quantity}\n"
        summary += f"Unit Price: ${self.current_product['price']:.2f}\n"
        summary += f"Total Amount: ${total_amount:.2f}\n"
        summary += f"Shipping Address: {self.shipping_address}\n"

        return summary

    async def handle_special_requests(self, message: str) -> str:
        """Handle special requests or questions"""
        prompt = f"""
        The user has a special request or question:
        "{message}"

        Generate a helpful response that:
        1. Addresses their specific request
        2. Maintains the current order context
        3. Guides them back to the ordering process if needed
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
            return response.choices[0].message.content

    async def handle_error_recovery(self, error_message: str) -> str:
        """Handle error recovery and provide guidance"""
        self.state = ConversationState.ERROR

        recovery_prompt = f"""
        An error occurred: "{error_message}"

        Generate a helpful response that:
        1. Apologizes for the error
        2. Explains what went wrong in simple terms
        3. Provides clear next steps
        4. Offers to start over if needed
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
            return response.choices[0].message.content

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