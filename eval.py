import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score ,recall_score
import os
import math
import cv2

if __name__ == '__main__':
    #classes_color_gray = [200, 100, 50, 150]
    #classes = ["Uterus", "Ovary", "Colon", "UterineTube"]
    predict_path = os.path.join("dataset", "result_fcn8s")#maskrcnn_15 #maskscroingrcnn_30
    ground_trust_path = os.path.join("dataset", "data_shape", "vaild_annot_mask")
    print("predict_path", predict_path)
    print("ground_trust_path", ground_trust_path)
    num = 0
    hist = np.zeros((5, 5))
    f1 = 0

    for index, i in enumerate(os.listdir(predict_path)):
        if i[-3:] == "png":
            num += 1
            print(index, len(os.listdir(predict_path)))
            print(os.path.join(ground_trust_path, i))
            file_string = os.path.join(ground_trust_path, i)[0:-12]
            print(file_string)
            pred_img = cv2.imread(os.path.join(predict_path, i), 0)
            gt_image = cv2.imread(file_string, 0)
            print(i+"   "+str(num))
            print(np.unique(pred_img.flatten()))
            print(confusion_matrix(gt_image.flatten(), pred_img.flatten(), labels=[0, 50, 100, 150, 200]))
            hist += confusion_matrix(gt_image.flatten(), pred_img.flatten(), labels=[0, 50, 100, 150, 200])
    print("=====================================done=========================================")    
    print(hist)
    acc = np.diag(hist).sum()/ hist.sum()
    classacc = np.diag(hist) / hist.sum(1)
    macc = np.nanmean(np.diag(hist) / hist.sum(1))
    miu = np.nanmean(np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist)))
    print("pixelAccuracy:", acc)
    print("classPixelAccuracy", classacc)
    print("meanPixelAccuracy", macc)
    print("meanIntersectionOverUnion", miu)
    f = open('result_fcn8s.txt', 'w')
    f.write(str("Average Pixel Accuracy | classPixelAccuracy | Average Mean Accuracy | Average Mean IU  \n"))
    f.write(str(acc) + " | ")
    f.write(str(classacc) + " | ")
    f.write(str(macc) + " | ")
    f.write(str(miu) + " | \n")
    f.close() 
