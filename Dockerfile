FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install numpy
EXPOSE 8080
CMD ["python3", "-m", "core.inference.server", "--port", "8080"]
