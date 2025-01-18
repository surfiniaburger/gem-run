# Use the official Python image from the Docker Hub
FROM python:3.10-slim

# Set the working directory in the container to '/app'
WORKDIR /app

# Copy the 'pyproject.toml' and 'poetry.lock' (if available) to '/app/'
COPY pyproject.toml poetry.lock* /app/

# Install Poetry
RUN pip install poetry

# Configure Poetry: Do not create a virtual environment and do not ask any interactive question
RUN poetry config virtualenvs.create false \
  && poetry config installer.parallel false

# Install dependencies using Poetry
RUN poetry install --no-dev

# Copy the rest of your app's source code from your host to your image filesystem.
COPY . /app

# Adjust WORKDIR to the directory containing 'gem.py'
WORKDIR /app/mlb_fan_highlights/src

# Expose the port Streamlit app runs on
EXPOSE 8501

# Command to run the app
CMD ["streamlit", "run", "gem.py"]
