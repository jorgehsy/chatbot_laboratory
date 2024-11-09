from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from contextlib import contextmanager
from typing import Generator, List, Dict, Optional, Tuple
from datetime import datetime

from .models import Base, Customer, Product, Order, OrderItem, OrderStatus


class DatabaseManager:
    def __init__(self, database_url: str):
        self.engine = create_engine(database_url)
        self.SessionLocal = sessionmaker(bind=self.engine)
        Base.metadata.create_all(self.engine)

    @contextmanager
    def get_db(self) -> Generator:
        db = self.SessionLocal()
        try:
            yield db
        finally:
            db.close()

    async def get_customer(self, customer_id: int) -> Optional[Dict]:
        with self.get_db() as db:
            customer = db.query(Customer).filter(Customer.id == customer_id).first()
            if customer:
                return {
                    "id": customer.id,
                    "name": customer.name,
                    "email": customer.email,
                    "default_shipping_address": customer.default_shipping_address
                }
            return None

    async def get_product(self, product_id: int) -> Optional[Dict]:
        with self.get_db() as db:
            product = db.query(Product).filter(Product.id == product_id).first()
            if product:
                return {
                    "id": product.id,
                    "name": product.name,
                    "price": product.price,
                    "inventory_count": product.inventory_count
                }
            return None

    async def validate_inventory(self, product_id: int, quantity: int) -> Tuple[bool, str]:
        with self.get_db() as db:
            product = db.query(Product).filter(Product.id == product_id).first()
            if not product:
                return False, "Product not found"

            if product.inventory_count < quantity:
                return False, f"Insufficient inventory. Only {product.inventory_count} units available"

            if product.inventory_count - quantity < product.min_stock_level:
                return False, f"Order would put inventory below minimum stock level of {product.min_stock_level}"

            return True, "Inventory available"

    async def create_order(self, customer_id: int, items: List[Dict], shipping_address: str | None = None) -> Dict:
        with self.get_db() as db:
            try:
                # Get customer
                customer = db.query(Customer).filter(Customer.id == customer_id).first()
                if not customer:
                    raise ValueError("Customer not found")

                # Create order
                order = Order(
                    customer_id=customer_id,
                    shipping_address=shipping_address or customer.default_shipping_address,
                    status=OrderStatus.PENDING
                )
                db.add(order)
                db.flush()

                total_amount = 0
                # Process items
                for item in items:
                    product = db.query(Product).filter(Product.id == item["product_id"]).first()
                    if not product:
                        raise ValueError(f"Product {item['product_id']} not found")

                    # Validate inventory
                    if product.inventory_count < item["quantity"]:
                        raise ValueError(f"Insufficient inventory for {product.name}")

                    # Create order item
                    order_item = OrderItem(
                        order_id=order.id,
                        product_id=product.id,
                        quantity=item["quantity"],
                        unit_price=product.price
                    )
                    db.add(order_item)

                    # Update inventory
                    product.inventory_count -= item["quantity"]
                    total_amount += product.price * item["quantity"]

                order.total_amount = total_amount
                db.commit()

                return {
                    "order_id": order.id,
                    "total_amount": total_amount,
                    "status": order.status.value
                }

            except Exception as e:
                db.rollback()
                raise ValueError(f"Error creating order: {str(e)}")

    async def update_order_status(self, order_id: int, status: OrderStatus) -> bool:
        with self.get_db() as db:
            try:
                order = db.query(Order).filter(Order.id == order_id).first()
                if not order:
                    return False

                order.status = status
                db.commit()
                return True

            except SQLAlchemyError:
                db.rollback()
                return False

    async def get_order_history(self, customer_id: int) -> List[Dict]:
        with self.get_db() as db:
            orders = db.query(Order).filter(Order.customer_id == customer_id).all()
            return [{
                "id": order.id,
                "total_amount": order.total_amount,
                "status": order.status.value,
                "created_at": order.created_at
            } for order in orders]