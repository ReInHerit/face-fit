# FaceFit Web App

## How to upload your images
You can substitute your own images in place of the sample ones in the public/images folder.
The images must be in the 600x600 pixel format and must contain the entire face of the character. The character cannot be in profile. A three-quarter pose is accepted.  

They could be photos or painting. unrealistic drawings or paintings may produce unexpected results or not work completely
#### NAMING CONVENTION
The file names must follow the convention of those present in the folder, that is:  
>image + a two-digit number + extension  
> 
for example: image01.jpg, image03.jpg ... image99.jpg  
#### IMAGE INFORMATION
When uploading a new image, you must also update the ___painting_data.json___ file.  
This file is located in the public/json folder and contains specific information for each image. It has two fields for each image:  
1. ___description___, which includes a textual description of the image where a face replacement occurred, and will be included in the email sent at the end of the game session. The text in this field can be formatted using HTML tags such as <a href></a>, <br>.  

Example:  
```
"image01.jpg": {
        "description": "the Venus of the painting <a href=\"https://en.wikipedia.org/wiki/The_Birth_of_Venus\">the Birth of Venus</a> by <a href=\"https://en.wikipedia.org/wiki/Sandro_Botticelli\">Sandro Botticelli</a>. <br>Curiosity: Botticelli used <a href=\"https://en.wikipedia.org/wiki/Simonetta_Vespucci\">Simonetta Vespucci</a> as his muse for his Venus.",
        }
```
## Setup email address and password
To send emails with the morphed results, you will need to use (or set up) a Gmail account. Then, you will need to generate an App Password by following the instructions provided in the first point on this link: https://www.interviewqs.com/blog/py-email.
Once you have generated the App Password, you will need to enter it, along with your email address, into the ___public/json/password_gmail.json___ file. 
## Setup Google Analytics
To use Google Analytics, you will need to create a google analytics account and get the tracking ID. 
Then, you will need to enter it into the ___.env_template___ file and remove the **_template** part of the file name.

## How to manage docker
Once ready with the reference images and email setup you can create and launch the docker image.
#### Prerequisites
To get started, you will need to set up and run Docker on your operating system. If you are not familiar with Docker, please refer to the official documentation [here](https://docs.docker.com/).
Check that `public/js/start.sh` has executable rights on macOS/Linux machines.
#### Create your docker container image 
To build the image using the Dockerfile, open a terminal, navigate to the folder containing the Dockerfile and type:  
```
docker build -t my_webapp_name .
```  
The dot at the end specifies the current directory.  
#### Start your app container
Now that you have an image, you can run the application in a container. To do so, you will use the docker run command.  
```
docker run -v morphs-volume:/app/morphs:rw -p 8000:8000 -p 8050:8050 my_webapp_name
``` 
#### Play with the app
Now open a browser and go to the address:  
```
localhost:8000
```