# Use an official Python runtime as a base image
FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy the API script into the container
COPY API.py /app/

# Install dependencies
RUN pip install flask opencv-python numpy ultralytics pillow

# Expose the API port
EXPOSE 5000

# Run the API
CMD ["python", "API.py"]
