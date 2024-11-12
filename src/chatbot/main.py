from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional, Dict
import uvicorn
import logging
from datetime import datetime
import asyncio
from pydantic import BaseModel

from .config import get_settings, Settings
from .chat_manager import LLMEnhancedChatbot
from .models import OrderContext
from .database import DatabaseManager
from .utils import handle_error, measure_time, logger

# Initialize settings
settings = get_settings()

# Initialize FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description="AI-powered sales order processing system",
    version=settings.APP_VERSION,
    debug=settings.DEBUG
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class ChatMessage(BaseModel):
    message: str
    context_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    context_id: str
    created_at: datetime = datetime.now()

# Initialize chatbot
chatbot = LLMEnhancedChatbot(
    database_url=settings.DATABASE_URL,
    llm_provider=settings.LLM_PROVIDER
)

# Dependencies
async def verify_api_key(api_key: str = Header(None, alias=settings.API_KEY_HEADER)):
    if settings.ENVIRONMENT != "development" and api_key != settings.SECRET_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key

# Middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = datetime.now()
    response = await call_next(request)
    duration = (datetime.now() - start_time).total_seconds()
    logger.info(
        f"Method: {request.method} Path: {request.url.path} "
        f"Status: {response.status_code} Duration: {duration:.2f}s"
    )
    return response

# Error handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    error_info = handle_error(exc)
    logger.error(f"Global error handler: {error_info}")
    return JSONResponse(
        status_code=500,
        content={"error": error_info["error"], "detail": error_info["detail"]}
    )

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT
    }

# Chat endpoint
@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(verify_api_key)])
@measure_time
async def process_message(chat_message: ChatMessage):
    try:
        response = await chatbot.process_message(chat_message.message)
        return ChatResponse(
            response=response,
            context_id=chat_message.context_id or "new_context"
        )
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Order endpoints
@app.post("/orders", dependencies=[Depends(verify_api_key)])
async def create_order(order_context: OrderContext):
    try:
        order = await chatbot.db_manager.create_order(
            customer_id=order_context.customer_id,
            items=[{"product_id": item["product_id"], "quantity": item["quantity"]}
                   for item in order_context.items],
            shipping_address=order_context.shipping_address
        )
        return order
    except Exception as e:
        logger.error(f"Error creating order: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/orders/{order_id}", dependencies=[Depends(verify_api_key)])
async def get_order(order_id: int):
    try:
        order = await chatbot.db_manager.get_order_by_id(order_id)
        if not order:
            raise HTTPException(status_code=404, detail="Order not found")
        return order
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving order: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("Starting up the application...")
    # Initialize any additional services here
    if settings.REDIS_URL:
        # Initialize Redis connection
        pass

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down the application...")
    # Cleanup any resources here

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=settings.WORKERS,
        log_level=settings.LOG_LEVEL.lower()
    )