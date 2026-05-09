FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for torch and faiss
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Expose the FastAPI port
EXPOSE 8000

# Start the streaming API server
CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
