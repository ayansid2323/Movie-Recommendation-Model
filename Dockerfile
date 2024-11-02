FROM python:3.10-slim

WORKDIR /code

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r /code/requirements.txt

COPY ./main.py .
COPY ./app.py .
COPY ./data ./data

EXPOSE 8000
CMD ["python", "app.py"]
