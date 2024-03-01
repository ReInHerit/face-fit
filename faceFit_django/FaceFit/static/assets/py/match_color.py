import cv2
import numpy as np
from color_matcher import ColorMatcher
from color_matcher.normalizer import Normalizer
from FaceFit.static.assets.py.utils import preprocess_image, LAB_channels

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


def find_noise_scratches(img):  # De-noising
    dst = cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 5, 15)
    noise = cv2.subtract(img, dst)
    return dst, noise


def outliner(image):
    result_planes = []
    result_norm_planes = []
    rgb_planes = cv2.split(image)
    for plane in rgb_planes:
        try:
            dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
            bg_img = cv2.medianBlur(dilated_img, 21)
            # Perform further processing with bg_img if needed
            diff_img = 255 - cv2.absdiff(plane, bg_img)
            norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            result_planes.append(diff_img)
            result_norm_planes.append(norm_img)
        except Exception as e:
            print("Error occurred during processing:", e)
        else:
            print("Processing completed successfully!")

    result = cv2.merge(result_planes)
    result_norm = cv2.merge(result_norm_planes)
    return [result, result_norm]


def remove_shadows(img):
    if img.dtype != np.uint8:
        # Scale the image to the range [0, 255]
        scaled_img = cv2.convertScaleAbs(img, alpha=(255.0 / img.max()))

        # Convert the scaled image to uint8
        img = np.uint8(scaled_img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    cv2.imshow('blur',blur)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 1)
    cv2.imshow('thresh', thresh)
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=4)
    cv2.imshow('closing', closing)
    cv2.waitKey(0)
    # Find contours in the binary image (if needed for reference)
    contours, hierarchy = cv2.findContours(closing.copy(), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h < 20 or w < 20:
            continue
        mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
        mask[y:y + h, x:x + w] = 255
        masked_img = cv2.bitwise_and(img, img, mask=mask)
        mean_val = cv2.mean(masked_img, mask=mask)[0]
        img[mask == 0] = mean_val
    return img


def add_dark_pixels(bgr_img, l_channel):
    # Convert BGR image to float64
    bgr_img = bgr_img.astype(np.float64)

    # Normalize L channel to the range [0, 1]
    normalized_l = l_channel / 255.0

    # Set the threshold value to determine dark pixels
    threshold = 0.7  # Adjust the threshold value as needed

    # Create a mask for dark pixels based on the threshold
    dark_pixel_mask = normalized_l >= threshold
    blurred_dark_pixel_mask = cv2.GaussianBlur(dark_pixel_mask.astype(np.float64), (15, 15), 0)
    darkening_factor = 0.3

    # Darken the dark pixels in the image
    # bgr_img_darkened = bgr_img.copy()
    bgr_img_darkened = bgr_img * (
                blurred_dark_pixel_mask[:, :, np.newaxis]) + bgr_img * darkening_factor * (1-blurred_dark_pixel_mask[:,
                                                                                              :, np.newaxis])
    # Clip the output image to the valid range of 0-255
    np.clip(bgr_img_darkened, 0, 255, out=bgr_img_darkened)

    return bgr_img_darkened.astype(np.uint8)


def match_cumulative_cdf(source, template):
    """
    Return modified source array so that the cumulative density function of
    its values matches the cumulative density function of the template.
    """
    if source.dtype.kind == 'u':
        src_lookup = source.reshape(-1)
        src_counts = np.bincount(src_lookup)
        tmpl_counts = np.bincount(template.reshape(-1))

        # omit values where the count was 0
        tmpl_values = np.nonzero(tmpl_counts)[0]
        tmpl_counts = tmpl_counts[tmpl_values]
    else:
        src_values, src_lookup, src_counts = np.unique(source.reshape(-1),
                                                       return_inverse=True,
                                                       return_counts=True)
        tmpl_values, tmpl_counts = np.unique(template.reshape(-1),
                                             return_counts=True)

    # calculate normalized quantiles for each array
    src_quantiles = np.cumsum(src_counts) / source.size
    tmpl_quantiles = np.cumsum(tmpl_counts) / template.size

    interp_a_values = np.interp(src_quantiles, tmpl_quantiles, tmpl_values)
    return interp_a_values[src_lookup].reshape(source.shape)


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
        raise ValueError('Image and reference must have the same number of channels.')

    if channel_axis is not None:
        if image.shape[-1] != reference.shape[-1]:
            raise ValueError('Number of channels in the input image and reference image must match!')

        matched = np.empty_like(image)
        for channel in range(image.shape[-1]):
            matched[..., channel] = match_cumulative_cdf(image[..., channel], reference[..., channel])
    else:
        matched = match_cumulative_cdf(image, reference)

    if matched.dtype.kind == 'f':
        out_dtype = supported_float_type(image.dtype)
        matched = matched.astype(out_dtype, copy=False)
    return matched


def matching_color(img_ref, img_src):
    print('matching color', img_ref.shape, img_src.shape, img_ref.dtype, img_src.dtype)
    img_src = preprocess_image(img_src)
    img_ref = preprocess_image(img_ref)

    outline = outliner(img_src)[1]
    outline_L, outline_A, outline_B = LAB_channels(outline)

    cm = ColorMatcher()
    img_out3 = cm.transfer(src=img_src, ref=img_ref, method='mkl')
    img_out3 = Normalizer(img_out3).uint8_norm()

    cc = match_histograms(img_out3, img_ref, channel_axis=2)
    cc = add_dark_pixels(cc, outline_L)
    return cc
