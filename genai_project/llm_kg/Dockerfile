FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY llm_kg/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY llm_kg/llm_deploy.py /app/llm_deploy.py
COPY llm_kg/llm_client.py /app/llm_client.py

EXPOSE 8000

CMD ["uvicorn", "llm_deploy:app", "--host", "0.0.0.0", "--port", "8000"]
