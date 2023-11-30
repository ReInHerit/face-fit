import cv2
import numpy as np

def import_image(path):
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise Exception(f'Image not found: {path}')
    if image.ndim == 2:
        image = np.expand_dims(image, axis=-1)
    return image


def view_image(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()


def view_images_together(images):
    for i, image in enumerate(images):
        cv2.imshow(f'image{i}', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def hls_channels(image):
    hls_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    h, l, s = cv2.split(hls_image)
    return h, l, s


def LAB_channels(image):
    l, a, b = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2LAB))
    return l, a, b


def hls_to_bgr(h_channel, s_channel, l_channel):
    return cv2.cvtColor(cv2.merge((h_channel, s_channel, l_channel)), cv2.COLOR_HLS2BGR)


def preprocess_image(image):
    # Check if the image dtype is not uint8
    if image.dtype != np.uint8:
        # Scale the image to the range [0, 255]
        scaled_img = cv2.convertScaleAbs(image, alpha=(255.0 / image.max()))

        # Convert the scaled image to uint8
        image = np.uint8(scaled_img)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        if image.shape[2] == 1 :
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        # Check the number of channels in the image
        if image.shape[2] == 3:
            # Image is either BGR or RGB
            # Check if the first channel is Red (indicating RGB)
            if image[:, :, 2].mean() > image[:, :, 0].mean():
                print("Image is in RGB color format")
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                print("Image is in BGR color format")
        else:
            print("Image does not have 3 channels, cannot determine color format")

    return image

