FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY codecompass ./codecompass
ENV OLLAMA_URL=http://host.docker.internal:11434
EXPOSE 8000
CMD ["uvicorn", "codecompass.api:app", "--host", "0.0.0.0", "--port", "8000"]
