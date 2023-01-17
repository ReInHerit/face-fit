# Base image for the Python server
FROM nikolaik/python-nodejs:python3.11-nodejs19

# Create a working directory
WORKDIR /app

# Install Python packages
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
RUN apt-get update && apt-get install -y python3-opencv


COPY package*.json ./
RUN npm install
#RUN npm ci --only=production
# Copy the rest of the code
COPY . .

# Run the server
ENV HOST 0.0.0.0
ENV VOLUME morphs-volume
CMD /app/public/js/start.sh
#CMD ["dumb-init", "node", "/app/index.js"]

EXPOSE 8050
EXPOSE 8000
