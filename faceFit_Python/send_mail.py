import smtplib
import imghdr
import os
import re
from email.message import EmailMessage

def send(attachments_list, mail_to_address):
    msg= EmailMessage()

    my_address ='fitface.unifi@gmail.com'   #sender address
    app_generated_password = 'iksrhpyebmjchesm'      #generated passcode

    msg["Subject"] ="The Email Subject"
    msg["From"]= my_address   #sender address

    msg["To"] = mail_to_address  # 'arkfil@gmail.com'         #reciver addresss
    msg.set_content('''Hello,
    This is a test mail.
    In this mail we are sending some attachments.
    The mail is sent using Python SMTP library.
    Thank You
    ''')

    for address in attachments_list:
        head, tail = os.path.split(address)
        with open(address, "rb") as file:     #open image file
            file_data = file.read()
            file_type = imghdr.what(file.name)

            print(tail)
            file_name = tail  # file.name

        print("File has been attached to the message body")
        msg.add_attachment(file_data, maintype="image",
                           subtype=file_type, filename= file_name)   #attach image file to msg

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