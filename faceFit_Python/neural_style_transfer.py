from matplotlib import gridspec
import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as tf_hub
import PIL
import os
import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

base_path = 'images/face.jpg'
style_path = 'images/FACE_08_thumb.jpg'


def load_image(image_path, image_size=(512, 256)):
    img = tf.io.decode_image(
      tf.io.read_file(image_path),
      channels=3, dtype=tf.float32)[tf.newaxis, ...]
    img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
    return img


def visualize(images, titles=('',), rows=False):
    noi = len(images)
    image_sizes = [image.shape[1] for image in images]
    w = (image_sizes[0] * 6) // 320
    plt.figure(figsize=(w * noi, w))
    grid_look = gridspec.GridSpec(1, noi, width_ratios=image_sizes)
    if rows == True:
        for i in range(noi):
            plt.subplot(grid_look[i])
            plt.imshow(images[i][0], aspect='equal')
            plt.axis('off')
            plt.title(titles[i])
            # plt.savefig("final.jpg")
    else:
        for i in range(noi):
            plt.subplot(grid_look[i])
            plt.imshow(images[i], aspect='equal')
            plt.axis('off')
            plt.title(titles[i])
            # plt.savefig("final.jpg")

    plt.show()


def export_image(tf_img):
    tf_img = tf_img*255
    tf_img = np.array(tf_img, dtype=np.uint8)
    if np.ndim(tf_img)>3:
        assert tf_img.shape[0] == 1
        img = tf_img[0]
    return PIL.Image.fromarray(img)



def print_style_transfer(or_img, st_img):
    image_size = (256, 512)
    or_norm = or_img/255
    cam_img = cv2.resize(or_norm, image_size)
    st_norm = st_img/255
    ref_img = cv2.resize(st_norm, image_size)
    cam_img = tf.convert_to_tensor(cam_img, dtype=tf.float32)
    ref_img = tf.convert_to_tensor(ref_img, dtype=tf.float32)
    assert isinstance(cam_img, tf.Tensor)
    assert isinstance(ref_img, tf.Tensor)
    cam_img = tf.expand_dims(cam_img, 0)
    ref_img = tf.expand_dims(ref_img, 0)
    style_image = tf.nn.avg_pool(ref_img, ksize=[3, 3], strides=[1, 1], padding='VALID')
    stylize_model = tf_hub.load('tf_model')

    results = stylize_model(tf.constant(cam_img), tf.constant(style_image))
    stylized_image_tf = results[0]
    # squeezing back the image to 3D
    image_tf_3D = tf.squeeze(stylized_image_tf)
    # Above is still a tensor. So we need to convert it to numpy. We do this by using tf session.
    image_numpy = image_tf_3D.numpy()
    # Since it is in float32 format, we need to convert it back to uint8.
    image_numpy = cv2.cvtColor(image_numpy, cv2.COLOR_RGB2BGR)
    image_numpy = cv2.resize(image_numpy, (or_img.shape[1],or_img.shape[0]))

    return image_numpy
# export_image(stylized_photo).save("my_stylized_photo.png")