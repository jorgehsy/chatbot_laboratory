# AI Sales Chatbot

An intelligent chatbot system for processing sales orders, built with Python, FastAPI, and AI integration.

## Features

- Natural language order processing
- Inventory management
- Bulk order handling
- Database integration
- Docker support
- Comprehensive testing suite

## Requirements

- Python 3.11+
- PostgreSQL 12+
- Docker & Docker Compose (optional)

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/yourusername/sales-chatbot.git
cd sales-chatbot
```

2. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. Run with Docker:
```bash
docker-compose up --build
```

Or run locally:
```bash
# Install dependencies
poetry install

# Initialize database
poetry run python scripts/init_db.py

# Run the application
poetry run python src/chatbot/main.py
```

## Development

1. Run tests:
```bash
poetry run pytest
```

2. Run with hot reload:
```bash
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up
```

## API Documentation

API documentation is available at `/docs` when running the server.

## License

MIT License - see LICENSE file for details