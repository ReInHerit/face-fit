import json
import os
import shutil
import smtplib
import uuid
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from .models import Reference
import cv2
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from djangoProject import settings
from FaceFit.static.assets.py.swap_faces import morph
from FaceFit.static.assets.py.utils import create_face_dict, extract_index, readb64, round_num
from FaceFit.static.assets.py import Face_Maker as F_obj


ref = []
ref_dict = []
ROOT_DIR = settings.BASE_DIR
assets_folder = os.path.join(ROOT_DIR, 'FaceFit', 'static', 'assets')
media_folder = os.path.join(ROOT_DIR, 'media')

images_folder = os.path.join(settings.MEDIA_ROOT, 'images')


ref = []
ref_dict = []

if os.getenv('HOST'):
    HOST = os.getenv('HOST')
else:
    HOST = 'localhost'

def home(request):
    refs = Reference.objects.all()
    print(refs)
    context = {
        'title': 'FaceFit',
        'ga_key': settings.GA_KEY,  # Replace with your Google Analytics key
        'data': refs,  # Add your data as needed
    }
    return render(request, 'FaceFit/index.html', context)
def policy(request):
    return render(request, 'FaceFit/policy.html')


def set_user(request):
    try:
        # Generate a temporary user ID using UUID
        user_id = str(uuid.uuid4())
        # Create the user folder
        user_folder = os.path.join(assets_folder, 'temp_folders', user_id)
        morphs_folder = os.path.join(user_folder, 'morphs')
        os.makedirs(morphs_folder, exist_ok=True)
        # images_folder = os.path.join(static_folder, 'assets/images')  # Replace with the actual path
        face_dict = create_face_dict(images_folder)

        # Update the global ref_dict with the generated face_dict
        global ref_dict
        ref_dict = face_dict

        return JsonResponse({'user_id': user_id, 'user_folder': user_folder})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


def get_dataset(request):
    print('Getting dataset')
    dataset = Reference.objects.values()
    dataset_list = list(dataset)
    global ref_dict
    # Extract only the necessary information
    simplified_dataset = [
        {
            'id': item['id'],
            'src': item['source'],
            'title': item['reference_title'],
            'text': item['reference_text'],
            # Add other fields as needed
        }
        for item in dataset_list
    ]
    try:
        ref_dict = []
        for idx, data in enumerate(simplified_dataset):
            ref_img = cv2.imread(os.path.join(media_folder, data['src']))
            p_face = F_obj.Face('ref')
            p_face.get_landmarks(ref_img)
            face_dict = {
                 'which': p_face.which,
                 'id': idx,
                 'src': data['src'],
                 'points': p_face.points,
                 'expression': [p_face.status['l_e'], p_face.status['r_e'], p_face.status['lips']],
                 'pix_points': p_face.pix_points,
                 'angles': [round_num(p_face.alpha) + 90, round_num(p_face.beta) + 90, round_num(p_face.gamma)],
                 'bb': {'xMin': p_face.bb_p1[0], 'xMax': p_face.bb_p2[0], 'yMin': p_face.bb_p1[1],
                        'yMax': p_face.bb_p2[1], 'width': p_face.delta_x, 'height': p_face.delta_y,
                        'center': [p_face.bb_p1[0] + round_num(p_face.delta_x / 2),
                                   p_face.bb_p2[0] + round_num(p_face.delta_y / 2)]},
                 'ref_text': data['text'],
                 }
            ref_dict.append(face_dict)
        print('REFERENCES INIT DONE')
        # print(ref_dict)
        return JsonResponse({'ref_dict': ref_dict}, status = 200)

    except Exception as e:
        # Handle any exceptions that may occur during resource initialization
        return JsonResponse({'error': str(e)}, status = 500)

@csrf_exempt
def morph_view(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            # Use your imported function
            data_img = data['c_face']
            user = data['user_id']
            user_folder = data['user_folder']
            selected = data['selected']
            r_obj = ref_dict[selected]
            c_image = readb64(data_img)
            c_image = cv2.flip(c_image, 1)
            c_obj = F_obj.Face('cam')
            c_obj.get_landmarks(c_image)
            head, file_name = os.path.split(r_obj['src'])
            r_obj['src'] = os.path.join(images_folder, file_name)
            # Morph the faces
            print('Morphing..')
            output = morph(c_obj, r_obj)
            numb = "0" + str(selected + 1) if selected <= 8 else str(selected + 1)
            morphed_file_name = 'morph_' + numb + '.png'
            morphs_folder = os.path.join(user_folder, 'morphs')
            os.makedirs(morphs_folder, exist_ok=True)
            print('Saving in user folder..', morphs_folder, morphed_file_name)
            path = os.path.join(morphs_folder, morphed_file_name)
            print('Path:', path)
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


def send_email(request):
    if request.method == 'POST':
        user_input = json.loads(request.body)
        print(user_input)
        send_to = user_input['mail']
        user_folder = user_input['user_folder']
        morphs_path = user_input['user_folder'] + '/morphs'
        try:
            # Your existing send_mail logic
            send_mail(send_to, user_folder)
            return JsonResponse({'answer': 'sent'})
        except Exception as e:
            print(f'Error sending mail: {e}')
            return JsonResponse({'error': 'Could not send mail'}, status=500)
    return JsonResponse({'error': 'Invalid request method'}, status=400)


def delete_morphs(request):
    if request.method == 'POST':
        user_input = json.loads(request.body)
        print('in view delete_morphs',user_input)
        # morphs_path = user_input['morphs_path'] + '/morphs'
        user_folder = user_input['morphs_path']
        # print(morphs_path)
        try:
            # Your existing send_mail logic
            del_user_data(user_folder)
            return JsonResponse({'answer': 'deleted'})
        except Exception as e:
            print(f'Error deleting morphs: {e}')
            return JsonResponse({'error': 'Could not delete morphs'}, status=500)
        return JsonResponse({'error': 'Invalid request method'}, status=400)


def send_mail(send_to, path):
    morphs_path = os.path.join(path, 'morphs')
    gmail_email = settings.GMAIL_EMAIL
    gmail_password = settings.GMAIL_PASSWORD
    print(gmail_email, gmail_password, path)
    files = os.listdir(morphs_path)
    morph_list = [{'filename': file, 'path': os.path.join(morphs_path, file)} for file in files]

    content = 'Hello,<br>Face-Fit App here. These are the results of your matches.<br>The characters in which you impersonated yourself are:<br>'

    for morph in morph_list:
        numb = extract_index(morph['filename']) + 1
        index = numb - 1
        if numb <= 9:
            numb = '0' + str(numb)
        description = ref_dict[index]['ref_text'] # painting_data.get(f'image{numb}.jpg', {}).get('description', '')
        content += f'<li>{description}</li>'

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
            # del_user_data(path)
            print('Email sent successfully!')
    except Exception as e:
        print(f'Error sending mail: {e}')


def del_user_data(path):
    try:
        shutil.rmtree(path)
        print(f"Successfully deleted the directory: {path}")
    except OSError as e:
        print(f"Error deleting directory {path}: {e}")
