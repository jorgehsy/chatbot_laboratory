from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Enum as SQLEnum
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.sql import func
from enum import Enum
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field

Base = declarative_base()


class OrderStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    CONFIRMED = "confirmed"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"


# Add OrderContext Pydantic model
class OrderContext(BaseModel):
    customer_id: Optional[int] = None
    items: List[dict] = Field(default_factory=list)
    shipping_address: Optional[str] = None
    total_amount: Optional[float] = None
    special_instructions: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    status: Optional[OrderStatus] = Field(default=OrderStatus.PENDING)


class Customer(Base):
    __tablename__ = 'customers'

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    email = Column(String(255), unique=True)
    default_shipping_address = Column(String(255))
    phone = Column(String(20))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    orders = relationship("Order", back_populates="customer")


class Product(Base):
    __tablename__ = 'products'

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    description = Column(String(500))
    price = Column(Float, nullable=False)
    inventory_count = Column(Integer, default=0)
    min_stock_level = Column(Integer, default=5)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    order_items = relationship("OrderItem", back_populates="product")


class Order(Base):
    __tablename__ = 'orders'

    id = Column(Integer, primary_key=True)
    customer_id = Column(Integer, ForeignKey('customers.id'))
    shipping_address = Column(String(255))
    total_amount = Column(Float)
    status = Column(SQLEnum(OrderStatus), default=OrderStatus.PENDING)
    special_instructions = Column(String(500))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    customer = relationship("Customer", back_populates="orders")
    items = relationship("OrderItem", back_populates="order", cascade="all, delete-orphan")


class OrderItem(Base):
    __tablename__ = 'order_items'

    id = Column(Integer, primary_key=True)
    order_id = Column(Integer, ForeignKey('orders.id'))
    product_id = Column(Integer, ForeignKey('products.id'))
    quantity = Column(Integer, nullable=False)
    unit_price = Column(Float, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    order = relationship("Order", back_populates="items")
    product = relationship("Product", back_populates="order_items")


# Pydantic models for API requests/responses
class OrderItemCreate(BaseModel):
    product_id: int
    quantity: int
    unit_price: Optional[float] = None


class OrderCreate(BaseModel):
    customer_id: int
    items: List[OrderItemCreate]
    shipping_address: Optional[str] = None
    special_instructions: Optional[str] = None


class CustomerCreate(BaseModel):
    name: str
    email: str
    default_shipping_address: Optional[str] = None
    phone: Optional[str] = None


class ProductCreate(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
    inventory_count: int
    min_stock_level: Optional[int] = None


# Add these for serialization of database models
class CustomerResponse(BaseModel):
    id: int
    name: str
    email: str
    default_shipping_address: Optional[str]
    phone: Optional[str]

    class Config:
        from_attributes = True


class ProductResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    price: float
    inventory_count: int
    min_stock_level: int

    class Config:
        from_attributes = True


class OrderItemResponse(BaseModel):
    id: int
    product_id: int
    quantity: int
    unit_price: float

    class Config:
        from_attributes = True


class OrderResponse(BaseModel):
    id: int
    customer_id: int
    shipping_address: str
    total_amount: float
    status: OrderStatus
    special_instructions: Optional[str]
    items: List[OrderItemResponse]

    class Config:
        from_attributes = True


# Export all models
__all__ = [
    'Base',
    'OrderStatus',
    'OrderContext',
    'Customer',
    'Product',
    'Order',
    'OrderItem',
    'OrderItemCreate',
    'OrderCreate',
    'CustomerCreate',
    'ProductCreate',
    'CustomerResponse',
    'ProductResponse',
    'OrderItemResponse',
    'OrderResponse'
]