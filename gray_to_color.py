import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython import embed


def bgr_to_hsv(bgr, depth=255):
    """bgr : np.ndarray, shape (3,)"""
    R_INDEX = 2
    G_INDEX = 1
    B_INDEX = 0
    bgr_norm = bgr/depth
    r = bgr_norm[R_INDEX]
    g = bgr_norm[G_INDEX]
    b = bgr_norm[B_INDEX]
    c_max = bgr_norm.max()
    c_min = bgr_norm.min()
    delta = c_max - c_min
    max_index = np.argmax(bgr_norm)
    
    h = 0
    tolerance = 0.0001
    if abs(delta) < tolerance:
        h = 0
    elif max_index == R_INDEX:
        h = (60*((g-b)/delta) + 360) % 360
    elif max_index == G_INDEX:
        h = (60*((b-r)/delta) + 120) % 360
    elif max_index == B_INDEX:
        h = (60*((r-g)/delta) + 240) % 360

    s = 0
    if abs(c_max) < tolerance:
        s = 0        
    else:
        s = delta/c_max * 100

    v = c_max * 100

    hsv = np.array([h,s,v])

    return hsv

def hsv_to_bgr(hsv, depth=255):
    hsv_norm = hsv / np.array([1, 100, 100])
    h = hsv_norm[0] % 360
    s = hsv_norm[1]
    v = hsv_norm[2]
    c = v*s
    x = c * (1 - abs((h/60)%2 - 1))
    m = np.expand_dims(v - c, -1)

    rgb_choices = np.array([[c, x, 0],
                            [x, c, 0],
                            [0, c, x],
                            [0, x, c],
                            [x, 0, c],
                            [c, 0, x]])
    choice = int(h/60)
    rgb_prime = rgb_choices[choice, :]
    rgb = (rgb_prime + m) * depth
    bgr = rgb[::-1] 
    return bgr


IMG_PATH = Path("images/2022-10-20_10H09M52S697MS_CHECK_ARTEFACT_grid_stack_i.png")
REF_IMG_PATH = Path("images/2022-10-20_10H09M52S697MS_grid_stack_i_color_ref.png")

image = cv2.imread(str(IMG_PATH), flags=cv2.IMREAD_ANYDEPTH)

plt.imshow(image, cmap='gray')

# gray_min = np.min(image)
# gray_max = np.max(image)
gray_max = 30000
gray_min = 0
gray_range = gray_max - gray_min

bgr_val = np.array([1000, 42000, 8500])

hsv_val = bgr_to_hsv(bgr_val, depth=65535)
bgr_calc = hsv_to_bgr(hsv_val, depth=65535)
print(bgr_val)
print(hsv_val)
print(bgr_calc)


H_LOW = 120
H_HIGH = 240
h_range = H_HIGH - H_LOW

hsv_image = np.ones(image.shape + (3,))*100
hue = ((image - gray_min) * (h_range/gray_range)) + H_LOW
black_tolerance = 1
black_mask = hue < black_tolerance + H_LOW

hsv_image[:,:,0] = hue
hsv_image[black_mask] = 0

image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

for i in tqdm(range(image_color.shape[0])):
    for j in range(image_color.shape[1]):
        image_color[i,j,:] = hsv_to_bgr(hsv_image[i,j,:])

# image_color = hsv_to_bgr(hsv_image)

plt.figure()
plt.imshow(image_color[:,:,::-1])
plt.show()