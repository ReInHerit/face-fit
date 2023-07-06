import cv2
import numpy as np
from color_matcher import ColorMatcher
from color_matcher.normalizer import Normalizer

new_float_type = {
    # preserved types
    np.float32().dtype.char: np.float32,
    np.float64().dtype.char: np.float64,
    np.complex64().dtype.char: np.complex64,
    np.complex128().dtype.char: np.complex128,
    # altered types
    np.float16().dtype.char: np.float32,
    'g': np.float64,      # np.float128 ; doesn't exist on windows
    'G': np.complex128,   # np.complex256 ; doesn't exist on windows
}


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
    cv2.destroyAllWindows()


def view_images_together(images):
    for i, image in enumerate(images):
        cv2.imshow(f'image{i}', image)
    cv2.waitKey(0)


def hls_channels(image):
    hls_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    h, l, s = cv2.split(hls_image)
    return h, l, s


def hls_to_bgr(h_channel, s_channel, l_channel):
    hls_image = cv2.merge((h_channel, l_channel, s_channel))
    bgr_image = cv2.cvtColor(hls_image, cv2.COLOR_HLS2BGR)
    return bgr_image


def preprocess_image(image):
    if image.dtype == np.uint8:
        return image

    return np.uint8(image * (255.0 / image.max()))


def find_noise_scratches(img):  # De-noising
    dst = cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 5, 15)
    noise = cv2.subtract(img, dst)
    return dst, noise


def outliner(image):
    result_planes = []
    result_norm_planes = []
    rgb_planes = cv2.split(image)
    kernel = np.ones((7, 7), np.uint8)
    for plane in rgb_planes:
        bg_img = None  # Define bg_img with a default value
        try:
            dilated_img = cv2.dilate(plane, kernel)
            bg_img = cv2.medianBlur(dilated_img, 21)
            if bg_img is not None:
                diff_img = 255 - cv2.absdiff(plane, bg_img)
                norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
                result_planes.append(diff_img)
                result_norm_planes.append(norm_img)
            # Perform further processing with bg_img if needed
        except Exception as e:
            print("Error occurred during processing:", e)
        else:
            print("Plane shape:", plane.shape)
            print("Plane data type:", plane.dtype)

        # Check if bg_img is assigned before calculating diff_img and norm_img


    result = cv2.merge(result_planes)
    result_norm = cv2.merge(result_norm_planes)
    return [result, result_norm]


def add_dark_pixels(bgr_img, l_channel):
    # Convert BGR image to float64
    bgr_img = bgr_img.astype(np.float64)

    # Normalize L channel to the range [0, 1]
    normalized_l = l_channel / 255.0

    # Set the threshold value to determine dark pixels
    threshold = 0.01  # Adjust the threshold value as needed

    # Create a mask for dark pixels based on the threshold
    dark_pixel_mask = normalized_l >= threshold

    # Multiply the BGR image with the dark pixel mask to darken dark pixels
    bgr_img_darkened = bgr_img * dark_pixel_mask[:, :, np.newaxis]

    # Clip the output image to the valid range of 0-255
    np.clip(bgr_img_darkened, 0, 255, out=bgr_img_darkened)

    return bgr_img_darkened.astype(np.uint8)


def remove_shadows(img):
    print('in remove shadows')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 1)
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=4)
    print('finding contours')
    contours, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h < 50 or w < 50:
            continue
        mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
        mask[y:y + h, x:x + w] = 255
        masked_img = cv2.bitwise_and(img, img, mask=mask)
        mean_val = cv2.mean(masked_img, mask=mask)[0]
        img[mask == 0] = mean_val

    return img


def match_cumulative_cdf(source, template):
    """
    Return modified source array so that the cumulative density function of its values matches the cumulative density
    function of the template.
    """
    if source.dtype.kind == 'u':
        src_counts = np.bincount(source.ravel(), minlength=256)
        tmpl_counts = np.bincount(template.ravel(), minlength=256)

        # omit values where the count was 0
        tmpl_values = np.nonzero(tmpl_counts)[0]
        tmpl_counts = tmpl_counts[tmpl_values]
    else:
        src_values, src_counts = np.unique(source.ravel(), return_counts=True)
        tmpl_values, tmpl_counts = np.unique(template.ravel(), return_counts=True)

    # calculate normalized quantiles for each array
    src_quantiles = np.cumsum(src_counts) / source.size
    tmpl_quantiles = np.cumsum(tmpl_counts) / template.size

    interp_a_values = np.interp(src_quantiles, tmpl_quantiles, tmpl_values)
    return interp_a_values[source].reshape(source.shape)



def supported_float_type(input_dtype, allow_complex=False):
    """Return an appropriate floating-point dtype for a given dtype.
    float32, float64, complex64, complex128 are preserved.
    float16 is promoted to float32.
    complex256 is demoted to complex128.
    Other types are cast to float64.

    ----------Parameters
    input_dtype : np.dtype or tuple of np.dtype, The input dtype. If a tuple of multiple dtypes is provided, each
        dtype is first converted to a supported floating point type and the final dtype is then determined by applying
        `np.result_type` on the sequence of supported floating point types.
    allow_complex : bool, optional, If False, raise a ValueError on complex-valued inputs.
    -------Returns
    float_type : dtype
        Floating-point dtype for the image. """

    if isinstance(input_dtype, tuple):
        return np.result_type(*(supported_float_type(d) for d in input_dtype))
    input_dtype = np.dtype(input_dtype)
    if not allow_complex and input_dtype.kind == 'c':
        raise ValueError("complex valued input is not supported")
    return new_float_type.get(input_dtype.char, np.float64)


def match_histograms(image, reference, *, channel_axis=None):
    """Adjust an image so that its cumulative histogram matches that of another.
    The adjustment is applied separately for each channel.
    ----------Parameters
    image : ndarray, Input image. Can be gray-scale or in color.
    reference : ndarray, Image to match histogram of. Must have the same number of channels as image.
    channel_axis : int or None, optional, If None, the image is assumed to be a grayscale (single channel) image.
        Otherwise, this parameter indicates which axis of the array corresponds to channels.
    ----------Returns
    matched : ndarray, Transformed input image.
    Raises
    ----------ValueError
        Thrown when the number of channels in the input image and the reference differ.
    ----------References
    .. [1] http://paulbourke.net/miscellaneous/equalisation/ """

    if image.ndim != reference.ndim:
        raise ValueError('Image and reference must have the same number '
                         'of channels.')

    if channel_axis is not None:
        if image.shape[-1] != reference.shape[-1]:
            raise ValueError('Number of channels in the input image and '
                             'reference image must match!')

        matched = np.empty_like(image)
        for channel in range(image.shape[-1]):
            matched[..., channel] = match_cumulative_cdf(image[..., channel], reference[..., channel])
    else:
        matched = match_cumulative_cdf(image, reference)

    if matched.dtype.kind == 'f':
        # output a float32 result when the input is float16 or float32
        out_dtype = supported_float_type(image.dtype)
        matched = matched.astype(out_dtype, copy=False)
    return matched



def matching_color(img_ref, img_src):
    img_src = remove_shadows(img_src)
    img_src = preprocess_image(img_src)
    img_ref = preprocess_image(img_ref)

    outline_l = outliner(img_src)[1][:, :, 0]
    cm = ColorMatcher()
    img_out = cm.transfer(src=img_src, ref=img_ref, method='mkl')
    img_out = Normalizer(img_out).uint8_norm()

    cc = match_histograms(img_out, img_ref)
    cc = add_dark_pixels(cc, outline_l)

    return cc
