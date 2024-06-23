import time
import cv2
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
                                              0.3,
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

def change_image(image_path):
    rotation_angle = 30  # degrees
    brightness_value = 50  # adjust as needed
    crop_values = (50, 70, 640, 640)  # replace x, y, width, height with specific values


    # Load the image
    image = cv2.imread(image_path)

    # Rotate the image
    center = (image.shape[1] // 2, image.shape[0] // 2)  # image center
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
    image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))

    # Adjust brightness
    image = cv2.convertScaleAbs(image, alpha=1, beta=brightness_value)

    # Crop the image
    x, y, w, h = crop_values
    image = image[y:y+h, x:x+w]
    image = cv2.resize(image, (640, 640))
    # Save or use the modified image
    cv2.imwrite('temp.jpg', image)  # Example: saving the modified image
    return 'temp.jpg'

validation.define_label_path(validation.labels_blue_path)
model_blue = YOLO(r"C:\Users\emrey\OneDrive\Masaüstü\BitirmeFinal\yolov8\models\blue-best.pt")
model_red = YOLO(r"C:\Users\emrey\OneDrive\Masaüstü\BitirmeFinal\yolov8\models\red-best.pt")
model_rgb = YOLO(r"C:\Users\emrey\OneDrive\Masaüstü\BitirmeFinal\yolov8\models\rgb-best.pt")
model_flower = YOLO(r"C:\Users\emrey\OneDrive\Masaüstü\BitirmeFinal\yolov8\models\flower-best.pt")
blue_images, red_images, rgb_images, flower_images, _ = validation.get_image_names()
prediction_dict = {}
total_time = 0

red_path = r"D:\rgb-red-blue_augmented\red\test\images\Adsız2.jpg"
blue_path = r"D:\rgb-red-blue_augmented\blue\test\images\Adsız2.jpg"
rgb_path = r"D:\rgb-red-blue_augmented\rgb\test\images\Adsız2.jpg"


start_time = time.time()
blue_results = model_blue(blue_path, save=True)
red_results = model_red(red_path, save=True)
rgb_results = model_rgb(rgb_path, save=True)
flower_results = model_flower(rgb_path, save=True)

elapsed_time = time.time() - start_time
total_time += elapsed_time

blue_preds = blue_results[0][0]
blue_rbox, blue_allbox = postprocess(blue_results[0][0], blue_results[0][1], blue_results[0][2],
                                        impath=blue_path)

red_preds = red_results[0][0]
red_rbox, red_allbox = postprocess(red_results[0][0], red_results[0][1], red_results[0][2], impath=red_path)

flower_preds = flower_results[0][0]
flower_rbox, flower_allbox = postprocess(flower_results[0][0], flower_results[0][1], flower_results[0][2], impath=rgb_path)

rgb_preds = rgb_results[0][0]
rgb_im = rgb_results[0][1]
rgb_im0s = rgb_results[0][2]
rgb_rbox, rgb_allbox = postprocess(rgb_results[0][0], rgb_results[0][1], rgb_results[0][2], impath=rgb_path)


prediction_dict[rgb_path.split(".")[0]] = {}
prediction_dict[rgb_path.split(".")[0]]["flower"] = flower_allbox

prediction_dict[rgb_path.split(".")[0]]["red"] = red_allbox
prediction_dict[rgb_path.split(".")[0]]["blue"] = blue_allbox
prediction_dict[rgb_path.split(".")[0]]["rgb"] = rgb_allbox


try:
    orig_labels = validation.get_label('Adsız2')
    predicts = validation.validate(prediction_dict, inference=True)
    image = cv2.imread(validation.rgb_path + "\\" + rgb_path)
    for box_color in predicts[rgb_path.split(".")[0]]:
        box = box_color[0]
        green = 255 if "rgb" in box_color[1] else 0
        blue = 255 if "blue" in box_color[1] else 0
        red = 255 if "red" in box_color[1] else 0
        if "flower" in box_color[1]:
            red = 120
            green = 125
            blue = 125
        image = cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (red, green, blue), 2)
    # for box in orig_labels:
    #     image = cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 0), 2)
    results_path = r"C:\Users\emrey\OneDrive\Masaüstü\BitirmeFinal\yolov8\results_single\\"
    cv2.imwrite(results_path, image)
    # print(predicts)
except Exception as e:
    print(e)
    pass

print(total_time / len(rgb_images))