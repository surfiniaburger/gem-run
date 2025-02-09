FROM python:3.12

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

EXPOSE 8080

WORKDIR /app

COPY . ./

# Add the source directory to PYTHONPATH
ENV PYTHONPATH="${PYTHONPATH}:/app/mlb_fan_highlights/src"

# Update CMD to use the correct path
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "mlb_fan_highlights.src.middleware:application"]