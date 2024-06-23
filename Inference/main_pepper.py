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

validation.define_label_path(validation.labels_blue_path)
model_blue = YOLO(r"C:\Users\emrey\OneDrive\Masaüstü\BitirmeFinal\yolov8\models\blue-best.pt")
model_red = YOLO(r"C:\Users\emrey\OneDrive\Masaüstü\BitirmeFinal\yolov8\models\red-best.pt")
model_rgb = YOLO(r"C:\Users\emrey\OneDrive\Masaüstü\BitirmeFinal\yolov8\models\rgb-best.pt")
blue_images, red_images, rgb_images, _, _ = validation.get_image_names()
prediction_dict = {}

for ix in range(len(rgb_images)):


    blue_results = model_blue(validation.blue_path + "\\" + blue_images[ix], save=True)
    blue_preds = blue_results[0][0]
    blue_rbox, blue_allbox = postprocess(blue_results[0][0], blue_results[0][1], blue_results[0][2],
                                         impath=blue_images[ix])
    
    red_results = model_red(validation.red_path + "\\" + red_images[ix], save=True)
    red_preds = red_results[0][0]
    red_rbox, red_allbox = postprocess(red_results[0][0], red_results[0][1], red_results[0][2], impath=red_images[ix])

    rgb_results = model_rgb(validation.rgb_path + "\\" + rgb_images[ix], save=True)
    rgb_preds = rgb_results[0][0]
    rgb_im = rgb_results[0][1]
    rgb_im0s = rgb_results[0][2]
    rgb_rbox, rgb_allbox = postprocess(rgb_results[0][0], rgb_results[0][1], rgb_results[0][2], impath=rgb_images[ix])
    prediction_dict[rgb_images[ix].split(".")[0]] = {}

    prediction_dict[rgb_images[ix].split(".")[0]]["red"] = red_allbox
    prediction_dict[rgb_images[ix].split(".")[0]]["blue"] = blue_allbox
    prediction_dict[rgb_images[ix].split(".")[0]]["rgb"] = rgb_allbox
    

    try:
        orig_labels = validation.get_label(rgb_images[ix].split(".")[0])
        predicts = validation.validate(prediction_dict)
        image = cv2.imread(validation.rgb_path + "\\" + rgb_images[ix])
        for box_color in predicts[rgb_images[ix].split(".")[0]]:
            box = box_color[0]
            green = 255 if "rgb" in box_color[1] else 0
            blue = 255 if "blue" in box_color[1] else 0
            red = 255 if "red" in box_color[1] else 0
            image = cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (red, green, blue), 2)
        for box in orig_labels:
            image = cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 0), 2)
        results_path = r"C:\Users\emrey\OneDrive\Masaüstü\BitirmeFinal\yolov8\results_pepper\\"
        cv2.imwrite(results_path + rgb_images[ix], image)
        # print(predicts)
    except Exception as e:
        print(e)
        pass
