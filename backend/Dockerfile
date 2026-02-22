FROM python:3.11-slim

WORKDIR /app

# Copy requirements
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY backend/ ./backend/

# Expose port
EXPOSE 8000

# Run command
# main.py is in backend/main.py
# We can run it as a module
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
