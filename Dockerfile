# Use the official Python image from the Docker Hub
FROM python:3.12

# Expose the port Streamlit app runs on
EXPOSE 8080

# Set the working directory in the container to '/app'
WORKDIR /app

# Copy the 'pyproject.toml' and 'poetry.lock' (if available) to '/app/'
COPY pyproject.toml poetry.lock* /app/

# Install Poetry
RUN pip install poetry

# Configure Poetry: Do not create a virtual environment and do not ask any interactive question
RUN poetry config virtualenvs.create false \
  && poetry config installer.parallel false

# Copy the rest of your app's source code from your host to your image filesystem.
COPY . /app

# Install dependencies using Poetry
RUN poetry install --no-interaction

# Adjust WORKDIR to the directory containing 'gem.py'
WORKDIR /app/mlb_fan_highlights/src


# Command to run the app
CMD ["streamlit", "run", "mlb_fan_highlights/src/gem.py", "--server.port=8080", "--server.address=0.0.0.0"]
