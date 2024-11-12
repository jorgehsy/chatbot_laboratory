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
        return f"""You are a helpful sales assistant bot that helps create purchase orders.
        Current State: {self.state.value}
        Order Context: {self.order_context.json() if self.order_context else "No active order"}

        Based on the current state, you should:

        {self._get_state_instructions()}

        Always maintain a professional and helpful tone.
        If you need clarification, ask specific questions.
        Guide the user through the order process step by step.
        """
    def _get_state_instructions(self) -> str:
        """Get specific instructions based on current state"""
        instructions = {
            ConversationState.INIT: """
                - Greet the user
                - Ask if they're ready to place an order
                - Move to GREETING state
            """,
            ConversationState.GREETING: """
                - Identify if user is a returning customer
                - Ask for customer information
                - Move to CUSTOMER_SELECTION state
            """,
            ConversationState.CUSTOMER_SELECTION: """
                - Validate customer information
                - If valid, move to CUSTOMER_CONFIRMATION
                - If invalid, ask for clarification
            """,
            ConversationState.CUSTOMER_CONFIRMATION: """
                - Confirm customer details
                - Move to ORDER_START state
            """,
            ConversationState.ORDER_START: """
                - Ask what products they'd like to order
                - Move to PRODUCT_SELECTION state
            """,
            ConversationState.PRODUCT_SELECTION: """
                - Help select specific products
                - Validate product availability
                - Move to QUANTITY_INPUT state
            """,
            ConversationState.QUANTITY_INPUT: """
                - Get desired quantity
                - Validate inventory availability
                - Ask if they want to add more products
            """,
            ConversationState.ADD_MORE_PRODUCTS: """
                - If yes, return to PRODUCT_SELECTION
                - If no, move to SHIPPING_ADDRESS
            """,
            ConversationState.SHIPPING_ADDRESS: """
                - Confirm or update shipping address
                - Move to SPECIAL_INSTRUCTIONS state
            """,
            ConversationState.SPECIAL_INSTRUCTIONS: """
                - Ask for any special instructions
                - Move to ORDER_SUMMARY state
            """,
            ConversationState.ORDER_SUMMARY: """
                - Show complete order summary
                - Move to PRICE_CONFIRMATION state
            """,
            ConversationState.PRICE_CONFIRMATION: """
                - Confirm pricing details
                - Move to PAYMENT_METHOD state
            """,
            ConversationState.PAYMENT_METHOD: """
                - Get payment method
                - Move to ORDER_CONFIRMATION state
            """,
            ConversationState.ORDER_CONFIRMATION: """
                - Get final confirmation
                - If confirmed, move to ORDER_PROCESSING
                - If not, allow modifications
            """,
            ConversationState.ORDER_PROCESSING: """
                - Process the order
                - Update inventory
                - Move to ORDER_COMPLETE state
            """,
            ConversationState.ORDER_COMPLETE: """
                - Show order confirmation
                - Provide order number
                - Ask if they need anything else
            """
        }
        return instructions.get(self.state, "Proceed with the conversation naturally.")

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
        """Generate order summary"""
        summary = "Order Summary:\n"

        if self.order_context.customer_id:
            customer = await self.db_manager.get_customer(self.order_context.customer_id)
            summary += f"Customer: {customer['name']}\n"

        summary += f"Shipping to: {self.order_context.shipping_address}\n\n"

        summary += "Products:\n"
        for item in self.order_context.items:
            subtotal = item['quantity'] * item['unit_price']
            summary += f"- {item['quantity']}x {item['product_name']} @ ${item['unit_price']} = ${subtotal}\n"

        summary += f"\nTotal Amount: ${self.order_context.total_amount:.2f}"

        if self.order_context.special_instructions:
            summary += f"\n\nSpecial Instructions: {self.order_context.special_instructions}"

        return summary

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
            content_response = parse_openai_content(response.choices[0].message.content)
            print ("CONTENT_PARSED", content_response)
            return safe_json_loads(content_response)

    def _confirms_price(self, message: str) -> bool:
        """Check if user confirms the price"""
        positive_responses = {'yes', 'yeah', 'sure', 'ok', 'okay', 'correct', 'y', 'proceed'}
        return message.lower().strip() in positive_responses

    def _confirms_order(self, message: str) -> bool:
        """Check if user confirms the order"""
        positive_responses = {'yes', 'yeah', 'sure', 'ok', 'okay', 'confirm', 'y', 'place order'}
        return message.lower().strip() in positive_responses

    async def _process_order(self) -> str:
        """Process the order before final confirmation"""
        try:
            # Validate all items one final time
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
                return "Some items are no longer available:\n" + "\n".join(invalid_items)

            # Validate shipping address
            if not self.order_context.shipping_address:
                self.state = ConversationState.ERROR
                return "Shipping address is missing. Please provide a valid address."

            # Calculate final totals
            self._update_total_amount()

            return f"Order validated. Total amount: ${self.order_context.total_amount:.2f}\nProceed with order creation?"

        except Exception as e:
            logger.error(f"Error processing order: {str(e)}")
            self.state = ConversationState.ERROR
            return f"Error processing order: {str(e)}"

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
        """Handle cases where clarification is needed"""
        previous_state = self.state
        self.state = ConversationState.CLARIFICATION

        clarification_prompts = {
            ConversationState.CUSTOMER_SELECTION:
                "Could you please provide your name and email address?",
            ConversationState.PRODUCT_SELECTION:
                "Could you specify which product you're interested in?",
            ConversationState.QUANTITY_INPUT:
                "Please specify how many units you'd like to order.",
            ConversationState.SHIPPING_ADDRESS:
                "Please provide a complete shipping address.",
        }

        response = clarification_prompts.get(
            previous_state,
            "Could you please clarify your request?"
        )

        self.state = previous_state
        return response

    async def _handle_modification(self, message: str, extracted_info: Dict) -> str:
        """Handle order modifications"""
        modification_type = extracted_info.get('entities', {}).get('modification_type')

        if modification_type == 'quantity':
            return await self._modify_quantity(extracted_info)
        elif modification_type == 'product':
            return await self._modify_product(extracted_info)
        elif modification_type == 'shipping':
            return await self._modify_shipping(extracted_info)
        else:
            return "What would you like to modify? (quantity, product, or shipping address)"

    async def _modify_quantity(self, extracted_info: Dict) -> str:
        """Modify quantity of an existing item"""
        product_id = extracted_info.get('entities', {}).get('product_id')
        new_quantity = extracted_info.get('entities', {}).get('quantity')

        if not product_id or not new_quantity:
            return "Please specify which product and the new quantity."

        for item in self.order_context.items:
            if item['product_id'] == product_id:
                is_valid, message = await self.db_manager.validate_inventory(
                    product_id,
                    new_quantity
                )
                if is_valid:
                    item['quantity'] = new_quantity
                    self._update_total_amount()
                    return f"Quantity updated. New total: ${self.order_context.total_amount:.2f}"
                return f"Cannot update quantity: {message}"

        return "Product not found in your order."

    async def _modify_product(self, extracted_info: Dict) -> str:
        """Modify or remove a product"""
        product_id = extracted_info.get('entities', {}).get('product_id')
        action = extracted_info.get('entities', {}).get('action', 'remove')

        if action == 'remove':
            self.order_context.items = [
                item for item in self.order_context.items
                if item['product_id'] != product_id
            ]
            self._update_total_amount()
            return f"Product removed. New total: ${self.order_context.total_amount:.2f}"

        return "Please specify what you'd like to do with the product."

    async def _modify_shipping(self, extracted_info: Dict) -> str:
        """Modify shipping address"""
        new_address = extracted_info.get('entities', {}).get('shipping_address')

        if new_address:
            self.order_context.shipping_address = new_address
            return "Shipping address updated."

        return "Please provide the new shipping address."

    def _get_current_state_message(self) -> str:
        """Get appropriate message for current state"""
        state_messages = {
            ConversationState.INIT:
                "Welcome! How can I help you today?",
            ConversationState.CUSTOMER_SELECTION:
                "Please provide your customer information.",
            ConversationState.PRODUCT_SELECTION:
                "What product would you like to order?",
            ConversationState.QUANTITY_INPUT:
                "How many units would you like?",
            ConversationState.SHIPPING_ADDRESS:
                "Where should we ship your order?",
            ConversationState.ORDER_SUMMARY:
                "Here's your order summary.",
            ConversationState.ORDER_CONFIRMATION:
                "Would you like to confirm this order?",
            ConversationState.ERROR:
                "I apologize, but there was an error. Would you like to try again?"
        }
        return state_messages.get(self.state, "How can I help you?")

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
        """Handle the conversation based on current state"""
        print("STATE ", self.state)
        try:
            if self.state == ConversationState.INIT:

                if customer := await self._identify_customer(extracted_info):
                    self.order_context.customer_id = customer.get("id")
                    self.state = ConversationState.CUSTOMER_CONFIRMATION
                    return f"Welcome back {customer.get('name')}! Would you like to place a new order?"
                else:
                    self.state = ConversationState.GREETING
                return "Welcome! I'm here to help you place an order. Are you a returning customer?"

            elif self.state == ConversationState.GREETING:
                # Handle customer identification
                if customer := await self._identify_customer(extracted_info):
                    self.order_context.customer_id = customer.get("id")
                    self.state = ConversationState.ORDER_START
                    return f"Welcome back {customer.get('name')}! Would you like to place a new order?"
                else:
                    self.state = ConversationState.CUSTOMER_SELECTION
                    return "Could you please provide your customer information?"

            elif self.state == ConversationState.CUSTOMER_CONFIRMATION:
                # Handle customer confirmation
                if extracted_info["suggested_next_state"]:
                    new_state = ConversationState(extracted_info["suggested_next_state"])
                    print("NEW_STATE ",new_state)
                    if self.is_valid_state_transition(new_state):
                        self.state = new_state
                        return "Awesome! What products would you like to order?"
                    else:
                        self.state = ConversationState.CUSTOMER_SELECTION
                        return "How can I Help you?"

                else:
                    self.state = ConversationState.CUSTOMER_SELECTION
                    return "Could you please provide your customer information?"

            elif self.state == ConversationState.CUSTOMER_SELECTION:
                # Handle new customer registration or lookup
                customer_info = await self._process_customer_info(extracted_info)
                if customer_info:
                    self.state = ConversationState.ORDER_START
                    return "I've found your information. Would you like to proceed with placing an order?"
                return "I couldn't find your information. Could you please provide more details?"

            elif self.state == ConversationState.ORDER_START:
                self.state = ConversationState.PRODUCT_SELECTION
                return "What products would you like to order?"

            elif self.state == ConversationState.PRODUCT_SELECTION:
                # Handle product selection
                if product := await self._process_product_selection(extracted_info):
                    self.state = ConversationState.QUANTITY_INPUT
                    return f"How many {product.get('name')} would you like to order?"
                return "I couldn't find that product. Could you please specify another product?"

            elif self.state == ConversationState.QUANTITY_INPUT:
                # Handle quantity input and validation
                if await self._validate_quantity(extracted_info):
                    self.state = ConversationState.ADD_MORE_PRODUCTS
                    return "Would you like to add more products to your order?"
                return "I'm sorry, that quantity is not available. Please try a different quantity."

            elif self.state == ConversationState.ADD_MORE_PRODUCTS:
                if self._wants_more_products(message):
                    self.state = ConversationState.PRODUCT_SELECTION
                    return "What other products would you like to add?"
                self.state = ConversationState.SHIPPING_ADDRESS
                return "Great! Should we use your default shipping address?"

            elif self.state == ConversationState.SHIPPING_ADDRESS:
                # Handle shipping address confirmation/update
                if await self._process_shipping_address(extracted_info):
                    self.state = ConversationState.SPECIAL_INSTRUCTIONS
                    return "Would you like to add any special instructions for this order?"
                return "Could you please provide a valid shipping address?"

            elif self.state == ConversationState.SPECIAL_INSTRUCTIONS:
                self._add_special_instructions(message)
                self.state = ConversationState.ORDER_SUMMARY
                return await self._generate_order_summary()

            elif self.state == ConversationState.ORDER_SUMMARY:
                self.state = ConversationState.PRICE_CONFIRMATION
                return "The total for your order will be ${:.2f}. Would you like to proceed?".format(
                    self.order_context.total_amount or 0
                )

            elif self.state == ConversationState.PRICE_CONFIRMATION:
                if self._confirms_price(message):
                    self.state = ConversationState.ORDER_CONFIRMATION
                    return "Would you like to confirm this order?"
                self.state = ConversationState.CANCEL
                return "Would you like to modify the order or cancel it?"

            elif self.state == ConversationState.ORDER_CONFIRMATION:
                if self._confirms_order(message):
                    self.state = ConversationState.ORDER_PROCESSING
                    return await self._process_order()
                self.state = ConversationState.CANCEL
                return "Order cancelled. Would you like to start over?"

            elif self.state == ConversationState.ORDER_PROCESSING:
                order_result = await self._finalize_order()
                self.state = ConversationState.ORDER_COMPLETE
                return f"Your order has been successfully placed! Order ID: {order_result['order_id']}"

            elif self.state == ConversationState.ERROR:
                self.state = ConversationState.INIT
                return "I apologize for the error. Would you like to start over?"

            else:
                return await self.generate_response(message, extracted_info)

        except Exception as e:
            logger.error(f"Error handling state {self.state}: {str(e)}")
            self.state = ConversationState.ERROR
            return f"I apologize, but something went wrong. Would you like to try again?"

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
            return f"I apologize, but I encountered an error: {str(e)}. Let's start over."

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
            content_response = parse_openai_content(response.choices[0].message.content)
            print ("CONTENT_PARSED", content_response)
            return safe_json_loads(content_response)

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
            content_response = parse_openai_content(response.choices[0].message.content)
            print ("CONTENT_PARSED", content_response)
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