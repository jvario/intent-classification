FROM python:3.12

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /server

COPY . .

# Upgrade pip to the latest version
RUN pip install --upgrade pip

RUN pip install -r requirements.txt

# Command to run the FastAPI app using uvicorn
CMD ["python", "server.py"]

# Expose the port that FastAPI will run on
EXPOSE 8080