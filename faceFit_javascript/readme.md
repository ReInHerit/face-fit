<h1>FaceFit Web App</h1>

<h2>Prerequisites</h2>
<h3>How to upload your images</h3>
You can substitute your own images in place of the sample ones in the public/images folder.
The images must be in the 600x600 pixel format and must contain the entire face of the character. The character cannot be in profile. A three-quarter pose is accepted.  

They could be photos or painting. unrealistic drawings or paintings may produce unexpected results or not work completely

<h4><li>NAMING CONVENTION</li></h4>
The file names must follow the convention of those present in the folder, that is:  
>image + a two-digit number + extension  
> 
for example: image01.jpg, image03.jpg ... image99.jpg  
<h4><li>IMAGE INFORMATION</li></h4>
When uploading a new image, you must also update the ___public/json/painting_data.json___ file.  
This file is located in the public/json folder and contains specific information for each image. It has two fields for each image:  
1. ___description___, which includes a textual description of the image where a face replacement occurred, and will be included in the email sent at the end of the game session. The text in this field can be formatted using HTML tags such as `<a href></a>`, `<br>`.  

Example:  
```
"image01.jpg": {
        "description": "the Venus of the painting <a href=\"https://en.wikipedia.org/wiki/The_Birth_of_Venus\">the Birth of Venus</a> by <a href=\"https://en.wikipedia.org/wiki/Sandro_Botticelli\">Sandro Botticelli</a>. <br>Curiosity: Botticelli used <a href=\"https://en.wikipedia.org/wiki/Simonetta_Vespucci\">Simonetta Vespucci</a> as his muse for his Venus.",
        }
```
<h3>Setup email address and password</h3>
To send emails with the morphed results, you will need to use (or set up) a Gmail account. Then, you will need to generate an App Password by following the instructions provided in the first point on this link: https://www.interviewqs.com/blog/py-email.

After generating the App Password, you will need to enter it, along with your email address, into a copy of the ___.env_template___ file. Additionally, make sure to remove the **_template** part of the file name, so the updated file should be named ___.env___. This configuration file, .env, will then be used by the application to authenticate with the email service.

<h3>Setup Google Analytics</h3>
To use Google Analytics, you will need to create a google analytics account and get the tracking ID. 
Then, you will need to enter it into the ___.env___ file.
<h3>Download models</h3>
You will need to download the models and save them in the ___public/models___ folder:
- [face_landmarker.tasks](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task)
- [hair_segmenter.tflite](https://storage.googleapis.com/mediapipe-models/image_segmenter/hair_segmenter/float32/latest/hair_segmenter.tflite)

<h2>How to manage docker</h2>
Once ready with the reference images and email setup you can create and launch the docker image.
<h3>Prerequisites</h3>
To get started, you will need to set up and run Docker on your operating system. If you are not familiar with Docker, please refer to the official documentation [here](https://docs.docker.com/).
Check that `public/js/start.sh` has executable rights on macOS/Linux machines.
<h3>Create your docker container image </h3>
To build the image using the Dockerfile, open a terminal, navigate to the folder containing the Dockerfile and type:  
```
docker build -t my_webapp_name .
```  
The dot at the end specifies the current directory.  
<h3>Start your app container</h3>
Now that you have an image, you can run the application in a container. To do so, you will use the docker run command.  
```
docker run -v morphs-volume:/app/morphs:rw -p 8000:8000 -p 8050:8050 my_webapp_name
``` 
<h3>Play with the app</h3>
Now open a browser and go to the address:  
```
localhost:8000
```