FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY retrain.py .

# Папка для модели и датасетов
RUN mkdir -p /data/datacollect

ENV MODEL_PATH=/data/model.pkl
ENV OASIS_API_KEY=your_secret_key_here

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
