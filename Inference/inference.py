import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils import ops
from ultralytics.engine.results import Results
import validation


def replace_negatives_with_zero(lst):
    for sublist in lst:
        for i in range(len(sublist)):
            if sublist[i] < 0:
                sublist[i] = 0
    return lst


def extract_last_two_columns(allpreds):
    result = []
    for pred in allpreds:
        last_two_columns = pred[:-2]
        result.append(last_two_columns)
    return result


def postprocess(preds, img, orig_imgs, impath):
    preds, allpreds = ops.non_max_suppression(preds,
                                              0.1,
                                              0.00,
                                              agnostic=False,
                                              max_det=300,
                                              classes=None
                                              )
    if len(allpreds) > 0:
        allpreds = extract_last_two_columns(allpreds[0])
        allpreds = replace_negatives_with_zero(allpreds)
    else:
        allpreds = []
    if not isinstance(orig_imgs, list):
        orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)
    results = []
    names = {0: "0"}
    for i, pred in enumerate(preds):
        orig_img = orig_imgs[i]
        pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        results.append(Results(orig_img, path=impath, names=names, boxes=pred))

    # results is list of boxes but only the boxes that are detected with threshold
    # allpreds is list of boxes but all of the boxes
    return results, allpreds

def process_rgb_infra_images(image_path1, image_path2):
    # Read the first image
    image1 = cv2.imread(image_path1)
    image1 = cv2.resize(image1, (640, 640))
    cv2.imwrite(image_path1, image1)
    if image1 is None:
        raise ValueError("Image 1 not found at the specified path")

    # Read the second image and convert to grayscale
    image2 = cv2.imread(image_path2)
    image2 = cv2.resize(image2, (640, 640))
    cv2.imwrite(image_path2, image2)
    if image2 is None:
        raise ValueError("Image 2 not found at the specified path")
    image2_gray = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
    gray_width, gray_height = image2_gray.shape
    image2_gray = image2_gray[22:gray_height, :]
    new_gray = np.zeros((gray_width, gray_height), np.uint8)
    y_offset = 11
    new_gray[y_offset:y_offset + gray_width - y_offset*2, 0:gray_height] = image2_gray
    image2_gray = new_gray

    # Ensure the images are the same size
    if image1.shape[:2] != image2_gray.shape[:2]:
        raise ValueError("The dimensions of the two images do not match")

    # Create the first new image (replace Red channel)
    new_image1 = image1.copy()
    new_image1[:, :, 2] = image2_gray  # Replace the Red channel

    # Create the second new image (replace Blue channel)
    new_image2 = image1.copy()
    new_image2[:, :, 0] = image2_gray  # Replace the Blue channel

    # Save the new images
    cv2.imwrite('red.jpg', new_image1)
    cv2.imwrite('blue.jpg', new_image2)
import timeit
start_time = timeit.default_timer()
model_blue = YOLO(r"C:\Users\emrey\OneDrive\Masaüstü\BitirmeFinal\yolov8\models\blue-best.pt")
model_red = YOLO(r"C:\Users\emrey\OneDrive\Masaüstü\BitirmeFinal\yolov8\models\red-best.pt")
model_rgb = YOLO(r"C:\Users\emrey\OneDrive\Masaüstü\BitirmeFinal\yolov8\models\rgb-best.pt")
model_flower = YOLO(r"C:\Users\emrey\OneDrive\Masaüstü\BitirmeFinal\yolov8\models\flower-best.pt")
prediction_dict = {}


image_path = r"C:\Users\emrey\OneDrive\Masaüstü\BitirmeFinal\yolov8\test\rwgb\FLIR_20231202_125939_936.jpg"
infra_path = r"C:\Users\emrey\OneDrive\Masaüstü\BitirmeFinal\yolov8\test\FLIR_20231202_125939_936.jpg"
process_rgb_infra_images(image_path, infra_path)


blue_results = model_blue(image_path, save=True)
blue_preds = blue_results[0][0]
blue_rbox, blue_allbox = postprocess(blue_results[0][0], blue_results[0][1], blue_results[0][2],
                                     impath="blue.jpg")

red_results = model_red(image_path, save=True)
red_preds = red_results[0][0]
red_rbox, red_allbox = postprocess(red_results[0][0], red_results[0][1], red_results[0][2], impath="red.jpg")

flower_results = model_flower(image_path, save=True)
flower_preds = flower_results[0][0]
flower_rbox, flower_allbox = postprocess(flower_results[0][0], flower_results[0][1], flower_results[0][2], impath=image_path)

rgb_results = model_rgb(image_path, save=True)
rgb_preds = rgb_results[0][0]
rgb_im = rgb_results[0][1]
rgb_im0s = rgb_results[0][2]
rgb_rbox, rgb_allbox = postprocess(rgb_results[0][0], rgb_results[0][1], rgb_results[0][2], impath=image_path)

prediction_dict[image_path] = {}
prediction_dict[image_path]["red"] = red_allbox
prediction_dict[image_path]["blue"] = blue_allbox
prediction_dict[image_path]["rgb"] = rgb_allbox
prediction_dict[image_path]["flower"] = flower_allbox

try:
    predicts = validation.validate(prediction_dict, inference=True)
    image = cv2.imread(image_path)
    for box_color in predicts[image_path]:
        box = box_color[0]
        green = 255 if "rgb" in box_color[1] else 0
        blue = 255 if "blue" in box_color[1] else 0
        red = 255 if "red" in box_color[1] else 0
        if "flower" in box_color[1]:
            red = 120
            green = 125
            blue = 125
        image = cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (blue, green, red), 2)
    cv2.imwrite("./result.jpg", image)
except Exception as e:
    print(e)
    pass

elapsed = timeit.default_timer() - start_time
print(elapsed, " ms is the execution time")