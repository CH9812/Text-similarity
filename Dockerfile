# Use official Python image (compatible with Python 3.12)
FROM python:3.12

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Flask app and model files
COPY PartB.py .
COPY similarity_model.pkl .


# Expose the port Flask will run on
EXPOSE 5000

# Command to run the Flask app
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
