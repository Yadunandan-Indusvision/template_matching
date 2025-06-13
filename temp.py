import cv2
import matplotlib.pyplot as plt
import os
from glob import glob

image_folder = r'C:\Users\Admin\Desktop\template_matching\dataset_front'
template_path = r"C:\Users\Admin\Desktop\template_matching\template.jpg"

template = cv2.imread(template_path, 0)
if template is None:
    raise FileNotFoundError(f"Template not found at {template_path}")

image_paths = glob(os.path.join(image_folder, '*.jpg')) 

methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED',
           'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED',
           'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

output_root = "results_front"

for img_path in image_paths:
    print(f"\nProcessing: {img_path}")
    img = cv2.imread(img_path, 0)
    image_name = os.path.basename(img_path)

    for meth in methods:
        img2 = img.copy()
        method = eval(meth)
        res = cv2.matchTemplate(img2, template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        h, w = template.shape
        bottom_right = (top_left[0] + w, top_left[1] + h)

        cv2.rectangle(img2, top_left, bottom_right, 255, 2)

        method_dir = os.path.join(output_root, meth)
        os.makedirs(method_dir, exist_ok=True)
        output_path = os.path.join(method_dir, image_name)

        cv2.imwrite(output_path, img2)
        print(f"Saved to: {output_path}")
