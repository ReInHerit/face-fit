import smtplib
import imghdr
import os
import re
from email.message import EmailMessage
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


# class Subscriptable:
#     def __class_getitem__(cls, item):
#         return cls._get_child_dict()[item]
#
#     @classmethod
#     def _get_child_dict(cls):
#         return {k: v for k, v in cls.__dict__.items() if not k.startswith('_')}
#
#
# class Attachment(Subscriptable):
#     def __init__(self, data, type, name):
#         self.data = data
#         self.type = type
#         self.name = name


def send(attachments_list, mail_to_address, descriptions):
    attachments = []
    msg = EmailMessage()
    # msg = MIMEMultipart("alternative")
    my_address ='fitface.unifi@gmail.com'   #sender address
    app_generated_password = 'iksrhpyebmjchesm'      #generated passcode

    msg["Subject"] = "Fit-Face Matches"
    msg["From"] = my_address   #sender address

    msg["To"] = mail_to_address  # 'arkfil@gmail.com'         #reciver addresss
    msg.add_header('Content-Type', 'text/html')
    content = """Hello,<br>
    Fit-Face App here. These are the results of your matches.<br>
    The character in which you impersonated yourself is:<br>"""

    # for file in attachments_list:
    data_to_attach = {}
    id = 0
    for address in attachments_list:
        head, tail = os.path.split(address)
        with open(address, "rb") as file:     #open image file
            file_data = file.read()
            file_type = imghdr.what(file.name)
            file_name = tail  # file.name
            data_to_attach = [file_data, file_type, file_name]
        attachments.append(data_to_attach)
        numb = tail[6:8]
        image = "image" + numb + ".jpg"
        content += '<li>' + str(descriptions[image] + "</li><br>")
    msg.set_payload(content)
          #attach image file to msg
    for attach in attachments:
        print(attach[0])
        print("File has been attached to the message body")
        msg.add_attachment(attach[0], maintype="image",
                           subtype=attach[1], filename=attach[2])

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:

        smtp.login(my_address,app_generated_password)   #login to gmail
        print("sending mail\\.....")
        smtp.send_message(msg)    #send mail
        print("mail has sent!")


def check_email_address(address):
  # Checks if the address match regular expression
  is_valid = re.search('^\w+@\w+.\w+$', address)
  # If there is a matching group
  if is_valid:
    return True
  else:
    print('It looks that provided mail is not in correct format. \n'
          'Please make sure that you have "@" and "." in your address \n'
          'and the length of your mail is at least 6 characters long')
    return False