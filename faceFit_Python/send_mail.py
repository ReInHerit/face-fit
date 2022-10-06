import smtplib
import imghdr
import os
# import re
from email.message import EmailMessage


def send(attachments_list, mail_to_address, descriptions):
    attachments = []
    msg = EmailMessage()
    my_address = 'fitface.unifi@gmail.com'
    app_generated_password = 'iksrhpyebmjchesm'

    msg["Subject"] = "Your Face-Fit Images !"
    msg["From"] = my_address

    msg["To"] = mail_to_address
    msg.add_header('Content-Type', 'text/html')
    content = """Hello,<br>
    Face-Fit App here. These are the results of your matches.<br>
    The characters in which you impersonated yourself are:<br>"""

    for address in attachments_list:
        head, tail = os.path.split(address)
        with open(address, "rb") as file:     # open image file
            file_data = file.read()
            file_type = imghdr.what(file.name)
            file_name = tail  # file.name
            data_to_attach = [file_data, file_type, file_name]
        attachments.append(data_to_attach)
        numb = tail[6:8]
        image = "image" + numb + ".jpg"
        content += '<li>' + str(descriptions[image] + "</li>")
    msg.set_payload(content.encode('utf8'))
    # attach image file to msg
    for attach in attachments:
        print("File has been attached to the message body")
        msg.add_attachment(attach[0], maintype="image",
                           subtype=attach[1], filename=attach[2])

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:

        smtp.login(my_address, app_generated_password)   # login to gmail
        print("sending mail\\.....")
        smtp.send_message(msg)    # send mail
        print("mail has been sent!")
