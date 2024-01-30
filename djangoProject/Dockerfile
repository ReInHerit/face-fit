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
#ARG GMAIL_EMAIL
#ARG GMAIL_PASSWORD
#ARG DJANGO_KEY
#ENV GMAIL_EMAIL=$GMAIL_EMAIL
#ENV GMAIL_PASSWORD=$GMAIL_PASSWORD
#ENV DJANGO_KEY=$DJANGO_KEY
#RUN echo "DJANGO_KEY: $DJANGO_KEY"
# Load environment variables from .env file
RUN ["python", "-c", "import dotenv; dotenv.load_dotenv(dotenv_path='.env')"]
# Run the application
ENTRYPOINT ["python", "manage.py"]
CMD ["runserver", "0.0.0.0:8000"]
#for heroku
#CMD gunicorn --bind 0.0.0.0:$PORT djangoProject.wsgi