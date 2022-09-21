import matplotlib.pyplot as plt
import numpy as np
from skimage import data
import cv2
from skimage import exposure
from skimage.exposure import match_histograms
imm1 = cv2.imread('../images/image01.jpg')
imm2 = cv2.imread('../images/image07.jpg')
imm1 = cv2.cvtColor(imm1, cv2.COLOR_BGR2RGB)
imm2 = cv2.cvtColor(imm2, cv2.COLOR_BGR2RGB)
reference = np.array(imm1, dtype=np.uint8)
image = np.array(imm2, dtype=np.uint8)
matched = match_histograms(image, reference, channel_axis=-1)

# test = match_histograms(matched, image, channel_axis=-1)

# fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3), sharex=True, sharey=True)
# for aa in (ax1, ax2, ax3):
#     aa.set_axis_off()



# ax1.imshow(image)
# ax1.set_title('Source')
# ax2.imshow(matched)
# ax2.set_title('Reference')
# ax3.imshow(test)
# ax3.set_title('Matched')
#
# plt.tight_layout()
# plt.show()