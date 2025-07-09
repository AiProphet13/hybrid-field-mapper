FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV FLASK_APP=main.py
CMD ["gunicorn", "--workers", "4", "--bind", "0.0.0.0:5000", "main:server"]
