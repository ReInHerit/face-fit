from smtplib import SMTP_SSL
from imghdr import what
from os.path import split as split_path
from json import load as load_json
from email.message import EmailMessage

with open('password_gmail.json', 'r') as f:
    gmail = load_json(f)


def extract_nbr(input_str):
    if input_str is None or input_str == '':
        return 0

    out_number = ''
    for ele in input_str:
        if ele.isdigit():
            out_number += ele
    return out_number


def send(attachments_list, mail_to_address, descriptions):
    attachments = []
    msg = EmailMessage()
    my_address = gmail['email']
    app_generated_password = gmail['password']

    msg["Subject"] = "Your Face-Fit Images !"
    msg["From"] = my_address

    msg["To"] = mail_to_address
    msg.add_header('Content-Type', 'text/html')
    content = """Hello,<br>
    Face-Fit App here. These are the results of your matches.<br>
    The characters in which you impersonated yourself are:<br>"""

    for address in attachments_list:
        head, tail = split_path(address)
        with open(address, "rb") as file:     # open image file
            file_data = file.read()
            file_type = what(file.name)
            file_name = tail
            data_to_attach = [file_data, file_type, file_name]
        attachments.append(data_to_attach)
        numb = extract_nbr(str(file_name))
        description = descriptions[''.join("image" + numb + ".jpg")]["description"]
        content += ''.join('<li>' + description + "</li>")
    msg.set_payload(content.encode('utf8'))
    # attach image file to msg
    for attach in attachments:
        print("File has been attached to the message body")
        msg.add_attachment(attach[0], maintype="image", subtype=attach[1], filename=attach[2])
    with SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(my_address, app_generated_password)   # login to gmail
        print("sending mail\\.....")
        smtp.send_message(msg)    # send mail
        print("mail has been sent!")
