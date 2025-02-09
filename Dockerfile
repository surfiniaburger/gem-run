FROM python:3.12

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

EXPOSE 8080

WORKDIR /app

COPY . ./

RUN pip install --no-cache-dir -r requirements.txt

# Add the source directory to PYTHONPATH
ENV PYTHONPATH="${PYTHONPATH}:/app/mlb_fan_highlights/src"

# Make sure we use the full path to gunicorn
CMD ["/usr/local/bin/gunicorn", "--bind", "0.0.0.0:8080", "mlb_fan_highlights.src.middleware:application"]