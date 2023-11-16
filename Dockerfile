# Use the official Python 3.9 image as the base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements files into the container
COPY requirements*.txt ./

# Install the required dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements-dev.txt
RUN pip install --no-cache-dir -r requirements-test.txt

# Copy the entire project directory into the container
COPY . .

# Expose the port your FastAPI app will run on (adjust as needed)
EXPOSE 80

# Command to run your FastAPI app
CMD ["uvicorn", "challenge.api:app", "--host", "0.0.0.0", "--port", "80"]
