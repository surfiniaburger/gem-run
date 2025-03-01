FROM python:3.12

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

EXPOSE 8080

WORKDIR /app

COPY . ./

RUN pip install --no-cache-dir -r requirements.txt

# Change the entrypoint for Flask
CMD gunicorn mlb_fan_highlights.src.middleware:application --bind 0.0.0.0:$PORT