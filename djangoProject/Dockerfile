# Use the official Python image as a base image
FROM python:3.11.5

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies
RUN apt-get update \
    && apt-get install -y libgl1-mesa-glx \
    && apt-get install -y npm

# Set the working directory in the container
WORKDIR /app
# Copy the package.json file to the container at /app
COPY package.json /app/

# Install npm packages
RUN npm install --force

# Copy the requirements file to the container at /app
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app/

# Expose the port the app runs on
EXPOSE $PORT

# Load environment variables from .env file
RUN ["python", "-c", "import dotenv; dotenv.load_dotenv(dotenv_path='/app/.env')"]
# Run the application
ENTRYPOINT ["python", "manage.py"]
CMD ["runserver", "0.0.0.0:8000"]
#for heroku
#CMD gunicorn --bind 0.0.0.0:$PORT djangoProject.wsgi