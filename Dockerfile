FROM python:3.8-slim
WORKDIR /app
COPY . .
EXPOSE 3000 

RUN pip install flask
CMD ["python3", "main.py"]
