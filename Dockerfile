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

# Add environment variables for debugging
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=8080
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Use debug mode and increased timeout
CMD ["/usr/local/bin/gunicorn", \
     "--bind", "0.0.0.0:8080", \
     "--timeout", "120", \
     "--workers", "1", \
     "--log-level", "debug", \
     "mlb_fan_highlights.src.middleware:application_with_error_handling"]