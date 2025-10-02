FROM python:3.12.4-slim
WORKDIR /app
COPY . .
EXPOSE 3000

RUN pip install -r requirements.txt
CMD ["python3", "main.py"]
