FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

ENV PORT=8080

CMD ["uvicorn", "challenge.api:app", "--host", "0.0.0.0", "--port", "8080"]
