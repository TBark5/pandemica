# PANDEMICA dashboard in a container.
#   docker build -t pandemica .
#   docker run -p 8501:8501 pandemica      -> http://localhost:8501
FROM python:3.14-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
