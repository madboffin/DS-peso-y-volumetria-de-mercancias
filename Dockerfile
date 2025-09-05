FROM python:3.13-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
# RUN pip install -e .

EXPOSE 8000
CMD ["python", "cli/main.py"]
