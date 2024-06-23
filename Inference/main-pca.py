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

validation.define_label_path(validation.labels_pca_path)
model_pca = YOLO(r"C:\Users\emrey\OneDrive\Masaüstü\BitirmeFinal\yolov8\models\pca-best.pt")
# model_blue = YOLO(r"C:\Users\emrey\OneDrive\Masaüstü\BitirmeFinal\yolov8\models\blue-best.pt")
# model_red = YOLO(r"C:\Users\emrey\OneDrive\Masaüstü\BitirmeFinal\yolov8\models\red-best.pt")
# model_rgb = YOLO(r"C:\Users\emrey\OneDrive\Masaüstü\BitirmeFinal\yolov8\models\rgb-best.pt")
# model_flower = YOLO(r"C:\Users\emrey\OneDrive\Masaüstü\BitirmeFinal\yolov8\models\flower-best.pt")
blue_images, red_images, rgb_images, flower_images, pca_images = validation.get_image_names()
prediction_dict = {}

for ix in range(len(pca_images)):
    image1 = cv2.imread(validation.pca_path + "\\" + pca_images[ix])
    image1 = cv2.resize(image1, (640, 640))
    cv2.imwrite(validation.pca_path + "\\" + pca_images[ix], image1)
    pca_results = model_pca(validation.pca_path + "\\" + pca_images[ix], save=True)
    pca_preds = pca_results[0][0]
    pca_rbox, pca_allbox = postprocess(pca_results[0][0], pca_results[0][1], pca_results[0][2],
                                         impath=pca_images[ix])


    prediction_dict[pca_images[ix].split(".")[0]] = {}
    prediction_dict[pca_images[ix].split(".")[0]]["pca"] = pca_allbox

    

    try:
        orig_labels = validation.get_label(pca_images[ix].split(".")[0])
        predicts = validation.validate(prediction_dict)
        image = cv2.imread(validation.pca_path + "\\" + pca_images[ix])
        for box_color in predicts[pca_images[ix].split(".")[0]]:
            box = box_color[0]
            green = 255 if "rgb" in box_color[1] else 0
            blue = 255 if "blue" in box_color[1] else 0
            red = 255 if "red" in box_color[1] else 0
            red = 255 if "flower" in box_color[1] else 0
            red = 255 if "pca" in box_color[1] else 0
            image = cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (red, green, blue), 2)
        for box in orig_labels:
            image = cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 0), 2)
        results_path = r"C:\Users\emrey\OneDrive\Masaüstü\BitirmeFinal\yolov8\results-pca\\"
        cv2.imwrite(results_path + pca_images[ix], image)
        # print(predicts)
    except Exception as e:
        print(e)
        pass

    """
    1. Her model içinlabel ve image pathleri yükedik
    2. Modelleri yükledik
    
    3. Her image için gerekli resmi yükleyip 640x640 resize ettik
    4. Bütün modellere sokuldu resimler
    5. Modellerin ciktisindan tüm prediction boxlari liste olarak aldik (postprocess değiştirildi)
    6. Confidence 0.3 altı olanlar elendi.

    7. Tüm modellerin prediction boxlarini bir dictionaryde store ettik. Bu dictioanry, hangi modelin hangi biberi tespit ettiğini anlamamız ve analiz etmemiz icin gereklidir.
    8. Tum boxlari renklerine gore %30 IoU ile merge ettik. Örneğin bir alanda bir kirmizi ve bir mavi box varsa tek bir mor box haline geldi. Tüm boxlar birbirilye kiyasladik. eger herhangi iki box birbiriyle %30 üzeri kesişiyora hem boxları tek box olarak kapsayici alanla hem de renkerini birlestirerek tek bir box haline donusturduk.
    9. Soncuclari oriijinal labellar ile kiyasladik ve precision - recall degerleri hesaplandi.
    10. sonuclari orijinal resim uzerine cizdirdik.

    Tüm boxlar birbirilye kiyasladik. eger herhangi iki box birbiriyle %30 üzeri kesişiyora hem boxları tek box olarak kapsayici alanla hem de renkerini birlestirerek tek bir box haline donusturduk.
    """
