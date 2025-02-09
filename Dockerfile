FROM python:3.12

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

EXPOSE 8080

WORKDIR /app

COPY . ./

RUN pip install --no-cache-dir -r requirements.txt

CMD gunicorn --bind :8080 --workers 1 --threads 8 mlb_fan_highlights.src.main:app