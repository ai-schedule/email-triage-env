FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install numpy pillow

CMD ["python", "test_env.py"]
