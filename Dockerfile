# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 5555 available to the world outside this container
EXPOSE 5555

# Define environment variable
ENV SERVER_HOST 0.0.0.0

# Run app.py when the container launches
CMD ["python", "./src/app.py"]