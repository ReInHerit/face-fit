import json
import os
import shutil
import smtplib
import uuid
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from django.templatetags.static import static

from .models import Reference
import cv2
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from djangoProject import settings
from static.assets.py.swap_faces import morph
from static.assets.py.utils import create_face_dict, extract_index, readb64, round_num
from static.assets.py import Face_Maker as F_obj


ref = []
ref_dict = []
ROOT_DIR = settings.BASE_DIR
# media_folder = os.path.join(ROOT_DIR, 'media')
references_folder = os.path.join(settings.STATIC_ROOT, 'assets', 'images', 'references')
# images_folder = os.path.join(settings.MEDIA_ROOT, 'images')


HOST = os.getenv('HOST', 'localhost')
PORT = os.getenv('PORT', '8000')
PROTOCOL = os.getenv('PROTOCOL', 'http')


def home(request):
    refs = Reference.objects.all()
    print(refs)
    for ref in refs:
        ref.source = static('assets/images/references/' + ref.source.name)
    context = {
        'title': 'FaceFit',
        'ga_key': settings.GA_KEY,  # Replace with your Google Analytics key
        'data': refs,  # Add your data as needed
    }
    return render(request, 'FaceFit/index.html', context)


def policy(request):
    context = {
        'GMAIL_EMAIL': settings.GMAIL_EMAIL,
    }
    return render(request, 'FaceFit/policy.html', context)
@csrf_exempt
def set_user(request):
    try:
        # Generate a temporary user ID using UUID
        user_id = str(uuid.uuid4())
        # Create the user folder
        user_folder = os.path.join(ROOT_DIR, 'static', 'assets', 'temp_folders', user_id)
        morphs_folder = os.path.join(user_folder, 'morphs')
        print('Creating user folder..', user_folder)
        os.makedirs(morphs_folder, exist_ok=True)
        global ref_dict
        ref_dict = create_face_dict(references_folder)

        return JsonResponse({'user_id': user_id, 'user_folder': user_folder})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
def get_dataset(request):
    try:
        global ref_dict

        data = json.loads(request.body)
        index = data.get('index', 0)  # Get the index from the request, default to 0 if not provided

        dataset = Reference.objects.values()
        dataset_list = list(dataset)

        # Get the data for the specified index
        data = dataset_list[index]
        ref_dict[index]['ref_text'] = data['reference_text']
        return JsonResponse({'ref_dict': ref_dict[index]}, status=200)

    except Exception as e:
        # Handle any exceptions that may occur during resource initialization
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
def morph_view(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            # Use your imported function
            data_img = data['c_face']
            # user = data['user_id']
            user_folder = data['user_folder']
            selected = data['selected']
            print('Selected:', selected, 'User folder:', user_folder, 'References folder:', references_folder)
            r_obj = ref_dict[selected]
            c_image = readb64(data_img)
            c_image = cv2.flip(c_image, 1)
            c_obj = F_obj.Face('cam')
            c_obj.get_landmarks(c_image)
            head, file_name = os.path.split(r_obj['src'])
            r_obj['src'] = os.path.join(references_folder, file_name)
            # Morph the faces
            print('Morphing..')
            output = morph(c_obj, r_obj)
            numb = "0" + str(selected + 1) if selected <= 8 else str(selected + 1)
            morphed_file_name = 'morph_' + numb + '.png'
            morphs_folder = os.path.join(user_folder, 'morphs')
            os.makedirs(morphs_folder, exist_ok=True)
            print('Saving in user folder..', morphs_folder, morphed_file_name)
            path = os.path.join(morphs_folder, morphed_file_name)
            write = cv2.imwrite(path, output)
            if write:
                print('Saved')
                return JsonResponse({'file_name': morphed_file_name}, status=200)
            else:
                print('Failed to save')
                return JsonResponse({'status': 'error', 'message': 'Failed to save'}, status=500)

        except json.JSONDecodeError:
            response_data = {'status': 'error', 'message': 'Invalid JSON data'}
            return JsonResponse(response_data, status=400)

    else:
        return JsonResponse({'status': 'error', 'message': 'Method not allowed'}, status=405)

def get_dataset_length(request):
    dataset_length = Reference.objects.count()
    return JsonResponse({'datasetLength': dataset_length})
@csrf_exempt
def send_email(request):
    if request.method == 'POST':
        user_input = json.loads(request.body)
        print(user_input)
        send_to = user_input['mail']
        user_folder = user_input['user_folder']
        # morphs_path = user_input['user_folder'] + '/morphs'
        try:
            # Your existing send_mail logic
            send_mail(send_to, user_folder)
            return JsonResponse({'answer': 'sent'})
        except Exception as e:
            print(f'Error sending mail: {e}')
            return JsonResponse({'error': 'Could not send mail'}, status=500)
    return JsonResponse({'error': 'Invalid request method'}, status=400)


@csrf_exempt
def delete_morphs(request):
    if request.method == 'POST':
        user_input = json.loads(request.body)
        user_folder = user_input['morphs_path']
        try:
            del_user_data(user_folder)
            return JsonResponse({'answer': 'deleted'})
        except Exception as e:
            print(f'Error deleting morphs: {e}')
            return JsonResponse({'error': 'Could not delete morphs'}, status=500)
        # return JsonResponse({'error': 'Invalid request method'}, status=400)


def send_mail(send_to, path):
    morphs_path = os.path.join(path, 'morphs')
    gmail_email = settings.GMAIL_EMAIL
    gmail_password = settings.GMAIL_PASSWORD
    files = os.listdir(morphs_path)
    morph_list = [{'filename': file, 'path': os.path.join(morphs_path, file)} for file in files]

    content = 'Hello,<br>Face-Fit App here. These are the results of your matches.<br>The characters in which you impersonated yourself are:<br>'

    for morph in morph_list:
        numb = extract_index(morph['filename']) + 1
        index = numb - 1
        if numb <= 9:
            numb = '0' + str(numb)
        description = ref_dict[index]['ref_text']
        content += f'<li>{description}</li>'

    privacy_policy_url = f"{PROTOCOL}://{HOST}:{PORT}/FaceFit/policy/"
    content += f'''
    <div style="font-size: 0.8em; color: #888;">
        <br>You received this e-mail because, while using Face-Fit, you requested that the results be e-mailed to you. 
        As specified in the privacy policy you accepted, the head poses captured have been used to create the attached 
        final images. No files or images are stored in our system, and this e-mail will be promptly removed from our 
        servers after the result images are generated and sent to you. We do not share your head pose images or personal 
        information with any third-party services. Please see the <a href="{privacy_policy_url}"> Privacy Policy</a> for more details.
    </div>
    '''
    msg = MIMEMultipart()
    msg['From'] = gmail_email
    msg['To'] = send_to
    msg['Subject'] = 'Your Face-Fit Images !'

    msg.attach(MIMEText(content, 'html'))

    for morph in morph_list:
        with open(morph['path'], 'rb') as file:
            part = MIMEApplication(file.read(), Name=os.path.basename(morph['path']))
            part['Content-Disposition'] = f'attachment; filename="{os.path.basename(morph["path"])}"'
            msg.attach(part)

    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(gmail_email, gmail_password)
            server.sendmail(gmail_email, send_to, msg.as_string())
            print('Email sent successfully!')
    except Exception as e:
        print(f'Error sending mail: {e}')


def del_user_data(path):
    try:
        shutil.rmtree(path)
        print(f"Successfully deleted the directory: {path}")
    except OSError as e:
        print(f"Error deleting directory {path}: {e}")
