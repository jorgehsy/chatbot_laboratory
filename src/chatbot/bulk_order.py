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
        """Validar inventario para múltiples productos simultáneamente"""
        validation_results = []

        with self.db_manager.get_db() as db:
            for item in items:
                product = db.query(Product).filter(Product.id == item.product_id).first()
                if not product:
                    validation_results.append(InventoryValidationResult(
                        is_valid=False,
                        message=f"Producto con ID {item.product_id} no encontrado",
                        available_quantity=0,
                        product_name="Desconocido",
                        product_id=item.product_id,
                        min_stock_level=0,
                        current_price=0.0
                    ))
                    continue

                is_valid = True
                message = "Disponible"

                # Verificar si la cantidad está disponible
                if product.inventory_count < item.quantity:
                    is_valid = False
                    message = f"Solo {product.inventory_count} unidades disponibles"

                # Verificar nivel mínimo de stock
                elif product.inventory_count - item.quantity < product.min_stock_level:
                    is_valid = False
                    message = f"El pedido dejaría el inventario por debajo del nivel mínimo de stock ({product.min_stock_level})"

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
        """Crear pedido con múltiples productos"""
        with self.db_manager.get_db() as db:
            try:
                # Validar todos los productos primero
                validation_results = await self.validate_bulk_order(context.items)
                invalid_items = [r for r in validation_results if not r.is_valid]

                if invalid_items:
                    error_messages = [
                        f"{r.product_name}: {r.message}" for r in invalid_items
                    ]
                    raise ValueError(f"Validación de inventario fallida:\n" + "\n".join(error_messages))

                # Crear pedido principal
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

                # Crear elementos del pedido y actualizar inventario
                for item in context.items:
                    product = db.query(Product).filter(Product.id == item.product_id).first()

                    order_item = OrderItem(
                        order_id=order.id,
                        product_id=item.product_id,
                        quantity=item.quantity,
                        unit_price=product.price
                    )

                    # Actualizar inventario
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
                raise ValueError(f"Error en la base de datos: {str(e)}")

    async def split_bulk_order(self, context: BulkOrderContext) -> List[Dict]:
        """Dividir un pedido masivo en múltiples pedidos según disponibilidad de inventario"""
        available_items = []
        backorder_items = []

        # Verificar inventario y dividir elementos
        validation_results = await self.validate_bulk_order(context.items)
        for item, validation in zip(context.items, validation_results):
            if validation.is_valid:
                available_items.append(item)
            else:
                # Calcular cantidad disponible
                available_qty = validation.available_quantity
                if available_qty > 0:
                    # Dividir el elemento
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
        # Crear pedido para elementos disponibles
        if available_items:
            available_context = BulkOrderContext(
                customer_id=context.customer_id,
                items=available_items,
                shipping_address=context.shipping_address,
                special_instructions=f"{context.special_instructions} (Pedido Parcial - Artículos Disponibles)"
            )
            try:
                order = await self.create_bulk_order(available_context)
                order["order_type"] = "disponible"
                orders.append(order)
            except ValueError as e:
                print(f"Error al crear pedido de artículos disponibles: {e}")

        # Crear pedido pendiente para elementos restantes
        if backorder_items:
            backorder_context = BulkOrderContext(
                customer_id=context.customer_id,
                items=backorder_items,
                shipping_address=context.shipping_address,
                special_instructions=f"{context.special_instructions} (Pedido Pendiente)"
            )
            # Almacenar pedido pendiente para procesamiento futuro
            backorder = await self.create_backorder(backorder_context)
            backorder["order_type"] = "pendiente"
            orders.append(backorder)

        return orders

    async def create_backorder(self, context: BulkOrderContext) -> Dict:
        """Crear un pedido pendiente para artículos que no pueden ser surtidos inmediatamente"""
        with self.db_manager.get_db() as db:
            try:
                # Crear registro de pedido pendiente
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

                # Crear elementos del pedido pendiente
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
                    "status": "pendiente"
                }

            except SQLAlchemyError as e:
                db.rollback()
                raise ValueError(f"Error en la base de datos al crear pedido pendiente: {str(e)}")

    async def get_bulk_order_summary(self, order_id: int) -> Dict:
        """Obtener resumen detallado de un pedido masivo"""
        with self.db_manager.get_db() as db:
            order = db.query(Order).filter(Order.id == order_id).first()
            if not order:
                raise ValueError("Pedido no encontrado")

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
        """Verificar estado de múltiples pedidos simultáneamente"""
        with self.db_manager.get_db() as db:
            orders = db.query(Order).filter(Order.id.in_(order_ids)).all()
            return [{
                "order_id": order.id,
                "status": order.status.value,
                "total_amount": order.total_amount,
                "created_at": order.created_at
            } for order in orders]

    async def update_bulk_order_status(self, order_id: int, new_status: OrderStatus) -> bool:
        """Actualizar el estado de un pedido masivo"""
        try:
            return await self.db_manager.update_order_status(order_id, new_status)
        except Exception as e:
            print(f"Error al actualizar el estado del pedido: {e}")
            return False