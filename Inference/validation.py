import os


labels_blue_path = r"D:\labels\labels"
labels_pca_path = r"D:\pca-rgba\pca-rgba\test\labels"
labels_flower_path = r"D:\Flower Detection\Flower Detection\test\labels"

label_dict = {}
def define_label_path(label_path):
    labels = os.listdir(label_path)
    
    for label in labels:
        label_txt = open(label_path + "\\" + label, "r")
        label_txt = label_txt.readlines()
        label_txt = [line.rstrip() for line in label_txt]
        label_txt = [line.split(" ")[1:] for line in label_txt]
        label_txt = [[float(num)*640 for num in line] for line in label_txt]
        label_dict[label.split(".")[0]] = [[x - w / 2, y - h / 2, x + w / 2, y + h / 2] for x, y, w, h in label_txt]

rgb_path = r"D:\rgb-red-blue_augmented\rgb\test\images"
rgb_images = os.listdir(rgb_path)
rgb_images.sort()

red_path = r"D:\rgb-red-blue_augmented\red\test\images"
red_images = os.listdir(red_path)
red_images.sort()

blue_path = r"D:\rgb-red-blue_augmented\blue\test\images"
blue_images = os.listdir(blue_path)
blue_images.sort()

flower_path = r"D:\Flower Detection\Flower Detection\test\images"
flower_images = os.listdir(flower_path)
flower_images.sort()

pca_path = r"D:\pca-rgba\pca-rgba\test\images"
pca_images = os.listdir(pca_path)
pca_images.sort()

def get_image_names():
    return blue_images, red_images, rgb_images, flower_images, pca_images

def get_label(image_name):
    return label_dict[image_name]

def calculate_iou(box1, box2):
    """Calculate the Intersection over Union (IoU) of two bounding boxes."""
    # Determine the coordinates of the intersection rectangle
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Calculate the area of the intersection rectangle
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate the area of both bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Compute the IoU
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

def merge_boxes1(box1, box2):
    """Merge two bounding boxes into one."""
    x_min = min(box1[0], box2[0])
    y_min = min(box1[1], box2[1])
    x_max = max(box1[2], box2[2])
    y_max = max(box1[3], box2[3])
    return [x_min, y_min, x_max, y_max]

def merge_boxes(box1, box1_colors, box2, box2_colors):
    """Merge two bounding boxes and their colors."""
    x_min = min(box1[0], box2[0])
    y_min = min(box1[1], box2[1])
    x_max = max(box1[2], box2[2])
    y_max = max(box1[3], box2[3])
    merged_colors = list(set(box1_colors + box2_colors))  # Combine and remove duplicates
    return [x_min, y_min, x_max, y_max], merged_colors


def merge_overlapping_boxes1(predictions, iou_threshold):
    """Merge overlapping boxes based on IoU threshold."""
    merged = []
    while predictions:
        box = predictions.pop(0)
        i = 0
        while i < len(predictions):
            if calculate_iou(box, predictions[i]) >= iou_threshold:
                box = merge_boxes(box, predictions.pop(i))
            else:
                i += 1
        merged.append(box)
    return merged

def merge_overlapping_boxes(predictions, iou_threshold):
    """Merge overlapping boxes based on IoU threshold, considering colors."""
    merged = []
    colors = []
    while predictions:
        box, box_colors = predictions.pop(0)
        i = 0
        while i < len(predictions):
            if calculate_iou(box, predictions[i][0]) >= iou_threshold:
                box, box_colors = merge_boxes(box, box_colors, *predictions.pop(i))
            else:
                i += 1
        merged.append(box)
        colors.append(box_colors)
    return list(zip(merged, colors))

def calculate_precision_recall_and_final_predictions(ground_truths, predictions, iou_threshold=0.3):
    """
    Calculate precision, recall, and return final predictions after discarding singular boxes.
    ground_truths: List of ground truth bounding boxes [x_min, y_min, x_max, y_max].
    predictions: List of predicted bounding boxes [x_min, y_min, x_max, y_max].
    """
    preds = predictions.copy()
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    final_predictions = []
    preds = merge_overlapping_boxes(preds, iou_threshold)

    # Check each predicted box for overlap with ground truths
    for pred in preds:
        final_predictions.append(pred)  # Keep this prediction
        if any(calculate_iou(pred[0], gt) >= iou_threshold for gt in ground_truths):
            true_positives += 1
        else:
            false_positives += 1

    # Check each ground truth box for being detected
    for gt in ground_truths:
        if not any(calculate_iou(gt, pred[0]) >= iou_threshold for pred in preds):
            false_negatives += 1

    return false_positives, true_positives, false_negatives, final_predictions

def validate(predictions_dict, inference=False):
    """
    Validate the model on the test set.
    predictions_dict is a dictionary mapping image names to lists of bounding boxes [x_min, y_min, x_max, y_max].
    Returns a dictionary of final predictions after discarding singular boxes.
    """
    false_positives, true_positives, false_negatives = 0, 0, 0
    final_predictions_dict = {}
    total_founded = 0
    total_existed = 0
    for image_name, color_preds in predictions_dict.items():
        preds = []
        for color, boxes in color_preds.items():
            preds.extend([(box, [color]) for box in boxes])
        if not inference:
            ground_truths = label_dict[image_name]
        else:
            ground_truths = []
        fp, tp, fn, final_preds = calculate_precision_recall_and_final_predictions(ground_truths, preds)
        false_positives += fp
        true_positives += tp
        false_negatives += fn
        final_predictions_dict[image_name] = final_preds
        total_founded += len(final_preds)
        total_existed += len(ground_truths)

    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
    if not inference:
        print(f"Precision: {precision}, Recall: {recall} TP:{true_positives} FP:{false_positives} FN:{false_negatives} Total founded: {total_founded}, Total ground truth: {total_existed}")
    else:
        total_flower = 0
        total_pepper = 0
        for items in final_predictions_dict.items():
            preds = items[1]
            for pred in preds:
                if "flower" in pred[1]:
                    total_flower += 1
                else:
                    total_pepper += 1
        print(f"Total flower: {total_flower}, Total pepper: {total_pepper}")
    return final_predictions_dict