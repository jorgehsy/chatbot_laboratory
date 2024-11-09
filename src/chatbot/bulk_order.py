from typing import List, Dict, Optional, Tuple
from datetime import datetime
from pydantic import BaseModel, Field
from sqlalchemy.exc import SQLAlchemyError
from .models import Order, OrderItem, OrderStatus, Product
from .database import DatabaseManager


class BulkOrderItem(BaseModel):
    product_id: int
    quantity: int
    unit_price: Optional[float] = None
    product_name: Optional[str] = None


class BulkOrderContext(BaseModel):
    customer_id: int
    items: List[BulkOrderItem] = Field(default_factory=list)
    shipping_address: Optional[str] = None
    special_instructions: Optional[str] = None
    total_amount: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.now)


class InventoryValidationResult(BaseModel):
    is_valid: bool
    message: str
    available_quantity: int
    product_name: str
    product_id: int
    min_stock_level: int
    current_price: float


class BulkOrderManager:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    async def validate_bulk_order(self, items: List[BulkOrderItem]) -> List[InventoryValidationResult]:
        """Validate inventory for multiple products simultaneously"""
        validation_results = []

        with self.db_manager.get_db() as db:
            for item in items:
                product = db.query(Product).filter(Product.id == item.product_id).first()
                if not product:
                    validation_results.append(InventoryValidationResult(
                        is_valid=False,
                        message=f"Product ID {item.product_id} not found",
                        available_quantity=0,
                        product_name="Unknown",
                        product_id=item.product_id,
                        min_stock_level=0,
                        current_price=0.0
                    ))
                    continue

                is_valid = True
                message = "Available"

                # Check if quantity is available
                if product.inventory_count < item.quantity:
                    is_valid = False
                    message = f"Only {product.inventory_count} units available"

                # Check minimum stock level
                elif product.inventory_count - item.quantity < product.min_stock_level:
                    is_valid = False
                    message = f"Order would put inventory below minimum stock level ({product.min_stock_level})"

                validation_results.append(InventoryValidationResult(
                    is_valid=is_valid,
                    message=message,
                    available_quantity=product.inventory_count,
                    product_name=product.name,
                    product_id=product.id,
                    min_stock_level=product.min_stock_level,
                    current_price=product.price
                ))

        return validation_results

    async def create_bulk_order(self, context: BulkOrderContext) -> Dict:
        """Create order with multiple products"""
        with self.db_manager.get_db() as db:
            try:
                # Validate all products first
                validation_results = await self.validate_bulk_order(context.items)
                invalid_items = [r for r in validation_results if not r.is_valid]

                if invalid_items:
                    error_messages = [
                        f"{r.product_name}: {r.message}" for r in invalid_items
                    ]
                    raise ValueError(f"Inventory validation failed:\n" + "\n".join(error_messages))

                # Create main order
                order = Order(
                    customer_id=context.customer_id,
                    shipping_address=context.shipping_address,
                    status=OrderStatus.PENDING,
                    special_instructions=context.special_instructions
                )
                db.add(order)
                db.flush()

                total_amount = 0
                order_items = []

                # Create order items and update inventory
                for item in context.items:
                    product = db.query(Product).filter(Product.id == item.product_id).first()

                    order_item = OrderItem(
                        order_id=order.id,
                        product_id=item.product_id,
                        quantity=item.quantity,
                        unit_price=product.price
                    )

                    # Update inventory
                    product.inventory_count -= item.quantity
                    total_amount += product.price * item.quantity

                    order_items.append(order_item)
                    db.add(order_item)

                order.total_amount = total_amount
                db.commit()

                return {
                    "order_id": order.id,
                    "total_amount": total_amount,
                    "items_count": len(order_items),
                    "status": OrderStatus.PENDING.value
                }

            except SQLAlchemyError as e:
                db.rollback()
                raise ValueError(f"Database error: {str(e)}")

    async def split_bulk_order(self, context: BulkOrderContext) -> List[Dict]:
        """Split a bulk order into multiple orders based on inventory availability"""
        available_items = []
        backorder_items = []

        # Check inventory and split items
        validation_results = await self.validate_bulk_order(context.items)
        for item, validation in zip(context.items, validation_results):
            if validation.is_valid:
                available_items.append(item)
            else:
                # Calculate available quantity
                available_qty = validation.available_quantity
                if available_qty > 0:
                    # Split the item
                    available_items.append(BulkOrderItem(
                        product_id=item.product_id,
                        quantity=available_qty
                    ))
                    backorder_items.append(BulkOrderItem(
                        product_id=item.product_id,
                        quantity=item.quantity - available_qty
                    ))
                else:
                    backorder_items.append(item)

        orders = []
        # Create order for available items
        if available_items:
            available_context = BulkOrderContext(
                customer_id=context.customer_id,
                items=available_items,
                shipping_address=context.shipping_address,
                special_instructions=f"{context.special_instructions} (Partial Order - Available Items)"
            )
            try:
                order = await self.create_bulk_order(available_context)
                order["order_type"] = "available"
                orders.append(order)
            except ValueError as e:
                print(f"Error creating available items order: {e}")

        # Create backorder for remaining items
        if backorder_items:
            backorder_context = BulkOrderContext(
                customer_id=context.customer_id,
                items=backorder_items,
                shipping_address=context.shipping_address,
                special_instructions=f"{context.special_instructions} (Backorder)"
            )
            # Store backorder for future processing
            backorder = await self.create_backorder(backorder_context)
            backorder["order_type"] = "backorder"
            orders.append(backorder)

        return orders

    async def create_backorder(self, context: BulkOrderContext) -> Dict:
        """Create a backorder for items that cannot be fulfilled immediately"""
        with self.db_manager.get_db() as db:
            try:
                # Create backorder record
                order = Order(
                    customer_id=context.customer_id,
                    shipping_address=context.shipping_address,
                    status=OrderStatus.PENDING,
                    special_instructions=context.special_instructions
                )
                db.add(order)
                db.flush()

                total_amount = 0
                order_items = []

                # Create backorder items
                for item in context.items:
                    product = db.query(Product).filter(Product.id == item.product_id).first()

                    order_item = OrderItem(
                        order_id=order.id,
                        product_id=item.product_id,
                        quantity=item.quantity,
                        unit_price=product.price
                    )

                    total_amount += product.price * item.quantity
                    order_items.append(order_item)
                    db.add(order_item)

                order.total_amount = total_amount
                db.commit()

                return {
                    "order_id": order.id,
                    "total_amount": total_amount,
                    "items_count": len(order_items),
                    "status": "backorder"
                }

            except SQLAlchemyError as e:
                db.rollback()
                raise ValueError(f"Database error creating backorder: {str(e)}")

    async def get_bulk_order_summary(self, order_id: int) -> Dict:
        """Get detailed summary of a bulk order"""
        with self.db_manager.get_db() as db:
            order = db.query(Order).filter(Order.id == order_id).first()
            if not order:
                raise ValueError("Order not found")

            items_detail = []
            for item in order.items:
                product = db.query(Product).filter(Product.id == item.product_id).first()
                items_detail.append({
                    "product_name": product.name,
                    "quantity": item.quantity,
                    "unit_price": item.unit_price,
                    "subtotal": item.quantity * item.unit_price
                })

            return {
                "order_id": order.id,
                "customer_id": order.customer_id,
                "status": order.status.value,
                "total_amount": order.total_amount,
                "shipping_address": order.shipping_address,
                "special_instructions": order.special_instructions,
                "created_at": order.created_at,
                "items": items_detail
            }

    async def check_bulk_order_status(self, order_ids: List[int]) -> List[Dict]:
        """Check status of multiple orders simultaneously"""
        with self.db_manager.get_db() as db:
            orders = db.query(Order).filter(Order.id.in_(order_ids)).all()
            return [{
                "order_id": order.id,
                "status": order.status.value,
                "total_amount": order.total_amount,
                "created_at": order.created_at
            } for order in orders]

    async def update_bulk_order_status(self, order_id: int, new_status: OrderStatus) -> bool:
        """Update the status of a bulk order"""
        try:
            return await self.db_manager.update_order_status(order_id, new_status)
        except Exception as e:
            print(f"Error updating order status: {e}")
            return False