FROM python:3.12

EXPOSE 8080

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt


COPY main.py .

CMD ["python", "main.py"]