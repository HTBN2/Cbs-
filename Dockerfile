FROM python:3.11-slim

# System libs needed by opencv-python-headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8080
EXPOSE 8080

CMD gunicorn server:app --workers 2 --threads 4 --timeout 120 --bind 0.0.0.0:$PORT
