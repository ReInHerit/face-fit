# FaceFit App

## How to run the App
* We assume that you already have the following software installed:
    * Anaconda: This is a distribution of Python and R that includes the conda package manager, which is used to create and manage environments. You can download and install Anaconda from the official website: https://www.anaconda.com/products/distribution/
    * Python >= 3.9: The code uses Python to run the app. Check the Python version on your OS.  
  If you have Anaconda installed, Python should also be installed as it comes with Anaconda.
* open a terminal or the command prompt and navigate to the base folder of the project
* create a new conda environment typing:  
  ```conda create -n myenv```
* activate the environment:  
  ```conda activate myenv```
* install the packages listed in requirements.txt:  
  ```pip install -r requirements.txt```
* upload your images and setup your email folowing the above instructions
* run the app with:  
  ```python FFApp.py```

## How to upload your images
You can substitute your own images in place of the sample ones in the images folder.
The images must be in the 600x600 pixel format and must contain the entire face of the character. The character cannot be in profile. A three-quarter pose is accepted.  

They could be photos or painting. unrealistic drawings or paintings may produce unexpected results or not work completely
#### NAMING CONVENTION
The file names must follow the convention of those present in the folder, that is:  
>image + a two-digit number + extension  
> 
for example: image01.jpg, image03.jpg ... image99.jpg  
#### IMAGE INFORMATION
When uploading a new image, you must also update the ___painting_data.json___ file.  
This file is located in the base folder of the project and contains specific information for each image. It has two fields for each image:  
1. ___description___, which includes a textual description of the image where a face replacement occurred, and will be included in the email sent at the end of the game session. The text in this field can be formatted using HTML tags such as <a href></a>, <br>.  
2. ___center_delta___, This field holds a pair of values (in pixels) that can be added or subtracted from the system's calculated values if there are discrepancies in the replacement of the user's face on the image. These discrepancies are more likely to occur in paintings. To activate or deactivate these corrections, you need uncomment or comment line 614 __FFApp.py__ file.  

Example:  
```
"image01.jpg": {
        "description": "the Venus of the painting <a href=\"https://en.wikipedia.org/wiki/The_Birth_of_Venus\">the Birth of Venus</a> by <a href=\"https://en.wikipedia.org/wiki/Sandro_Botticelli\">Sandro Botticelli</a>. <br>Curiosity: Botticelli used <a href=\"https://en.wikipedia.org/wiki/Simonetta_Vespucci\">Simonetta Vespucci</a> as his muse for his Venus.",
        "center_delta": [8, -13]
        }
```
## Setup email address and password
To send emails with the morphed results, you will need to use (or set up) a Gmail account.  
Then, you will need to generate an App Password by following the instructions provided in the first point on this link: https://www.interviewqs.com/blog/py-email. 
Once you have generated the App Password, you will need to enter it, along with your email address, into the ___password_gmail.json___ file. 
