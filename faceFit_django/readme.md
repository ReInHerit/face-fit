<h1>FaceFit Web App</h1>

<h2>Prerequisites</h2>

<h3>Setup email address and password</h3>
To send emails with the morphed results, you will need to use (or set up) a Gmail account. Then, you will need to generate an App Password by following the instructions provided in the first point on this link: https://www.interviewqs.com/blog/py-email.

After generating the App Password, you will need to enter it, along with your email address, into a copy of the ___.env_template___ file. Additionally, make sure to remove the **_template** part of the file name, so the updated file should be named ___.env___. This configuration file, .env, will then be used by the application to authenticate with the email service.

<h3>Setup Google Analytics</h3>
To use Google Analytics, you will need to create a google analytics account and get the tracking ID. 
Then, you will need to enter it into the ___.env___ file.

<h3>Setup a Django secret key</h3> 
To generate a Django secret key in the terminal, navigate to the "utils" folder and run the following command:

```
python getYourDjangoKey.py 
```

Copy and paste the generated key in the DJANGO_KEY field of the ___.env___ file.

<h3>Download models</h3>
You will need to download the models and save them in the ___FaceFit/static/assets/models___ folder:
- [face_landmarker.tasks](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task)
- [hair_segmenter.tflite](https://storage.googleapis.com/mediapipe-models/image_segmenter/hair_segmenter/float32/latest/hair_segmenter.tflite)

<h3>Create a Django admin account</h3>
<h4>Creating a superuser</h4>
In order to log into the admin site, you need a user account with Staff status enabled. In order to view and create records you also need this user to have permissions to manage all our objects. 

You can create a "superuser" account that has full access to the site and all needed permissions using manage.py.

Open the console and call the following command, in the same directory as manage.py, to create the superuser. 
You will be prompted to enter a username, email address, and strong password.
```
python manage.py createsuperuser
```
Once this command completes a new superuser will have been added to the database. Now restart the development server so we can test the login:
```
python3 manage.py runserver
```

<h4>Logging in and using the site</h4>
To login to the site, open the /admin URL (e.g. http://127.0.0.1:8000/admin) and enter your new superuser userid and password credentials (you'll be redirected to the login page, and then back to the /admin URL after you've entered your details).

<h3>How to upload your images</h3>
Substitute your images for the sample ones in the FaceFit/assets/public/images folder. Ensure that your images are in the 600x600 pixel format, feature the entire face of the character, and avoid a profile view (a three-quarter pose is acceptable). Photos or paintings are accepted, but unrealistic drawings may yield unexpected results.
<h4><li>NAMING CONVENTION</li></h4>
File names must follow this convention::  
>image + a two-digit number + extension  
> 
e.g.: image01.jpg, image03.jpg ... image99.jpg  
<h4><li>IMAGE INFORMATION</li></h4>
When uploading a new image, provide the following details::  
-- ___Reference text___: A textual description of the image where face replacement occurred. This text will be included in the email sent at the end of the game session. HTML tags such as <a href></a> and <br> can be used for formatting.

Example:  
```
"the Venus of the painting <a href=\"https://en.wikipedia.org/wiki/The_Birth_of_Venus\">the Birth of Venus</a> by <a href=\"https://en.wikipedia.org/wiki/Sandro_Botticelli\">Sandro Botticelli</a>. <br>Curiosity: Botticelli used <a href=\"https://en.wikipedia.org/wiki/Simonetta_Vespucci\">Simonetta Vespucci</a> as his muse for his Venus.",
        
```
<h4><li>IMAGE UPLOAD</li></h4>
To upload custom images to the server:
1. Access the Django admin site after logging in. 
2. Select the **References** option under the "FACEFIT" title to view the image database.
3. Delete existing sample images by choosing the "Delete selected references" option from the Actions dropdown menu. Click the "Go" button to confirm removal. 
4. If the list is now empty, add a new image by clicking on the **ADD REFERENCE** option. Input the text for the final email in the "Reference text" field and select the image in the "Source" field. The Reference title will be populated automatically. Click the "SAVE" button.

<h2>How to manage docker</h2>

<h3>Prerequisites</h3>
To begin, set up and run Docker on your operating system. If you are unfamiliar with Docker, refer to the official documentation [here](https://docs.docker.com/).
<h3>Create your docker container image </h3>
To build the image using the Dockerfile, open a terminal, navigate to the folder containing the Dockerfile and type:  
```
docker build -t facefit .
```  
The dot at the end specifies the current directory.  
<h3>Start your app container</h3>
Now that you have an image, run the application in a container using the following docker run command.  
```
docker run --env-file=.env -v morphs-volume:/app/morphs:rw -p 8000:8000 -p 8050:8050 facefit
``` 
<h3>Play with the app</h3>
Now open a browser and go to the address:  
```
localhost:8000
```