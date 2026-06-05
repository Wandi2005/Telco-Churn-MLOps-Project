FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY 7.Inference.py .

EXPOSE 5001 8000

CMD ["python", "7.Inference.py"]