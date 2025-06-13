import cv2
import os
from glob import glob
import numpy as np

image_folder = r'C:\Users\Admin\Desktop\template_matching\dataset_back'
image_paths = glob(os.path.join(image_folder, '*.jpg')) 
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED',
           'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED',
           'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
output_root = "results_back"
tolerance = 10.0

first_img = cv2.imread(image_paths[0], 0)
cv2.namedWindow("Select Template", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Select Template", 400, 700)
roi = cv2.selectROI("Select Template", first_img, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("Select Template")

x, y, w, h = roi
template = first_img[y:y+h, x:x+w]
print(f"Selected template region: x={x}, y={y}, w={w}, h={h}")

for img_path in image_paths:
    print(f"\nProcessing: {img_path}")
    img = cv2.imread(img_path, 0)
    image_name = os.path.basename(img_path)

    for meth in methods:
        img2 = img.copy()
        method = eval(meth)
        res = cv2.matchTemplate(img2, template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        top_left = min_loc if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED] else max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        cv2.rectangle(img2, top_left, bottom_right, 255, 2)

        distance = np.linalg.norm(np.array(top_left) - np.array([x, y]))
        result_msg = "Template matched" if distance <= tolerance else "Template not matched"
        print(f"{meth} → {result_msg}")

        method_dir = os.path.join(output_root, meth)
        os.makedirs(method_dir, exist_ok=True)
        output_path = os.path.join(method_dir, image_name)
        cv2.imwrite(output_path, img2)
