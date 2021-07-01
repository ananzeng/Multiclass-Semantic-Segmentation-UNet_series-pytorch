import os
import json
import cv2
import numpy as np

labels = ["circle", "square", "triangle", "star"]
labels_color = [(0, 255, 0), (255, 0, 0), (255, 255, 0), (0, 255, 255)]
def get_poly(annot_path): 
    with open(annot_path) as handle:
        data = json.load(handle)
    shape_dicts = data['shapes']
    return shape_dicts

def create_multi_masks(filename,shape_dicts, jsonname):
    cls = [x['label'] for x in shape_dicts]
    print(cls)
    poly = [np.array(x['points'], dtype=np.int32) for x in shape_dicts] 
    label2poly = dict(zip(cls, poly)) #
    blank = np.zeros(shape=(590, 845, 3), dtype=np.uint8)
    for i, label in enumerate(labels):
        if label in cls:
            cv2.fillPoly(blank, [label2poly[label]], labels_color[labels.index(label)])
    gt_image = cv2.imread(os.path.join(image_dirname, jsonname.split(".")[0] + ".png"))
    dst=cv2.addWeighted(gt_image,1,blank,0.5,0)
    cv2.imwrite(save_dirname + "/" + jsonname[0:-5] + ".png", dst)

annot_dirname = "dataset/data_shape/vaild_annot"
image_dirname = "dataset/data_shape/vaild_image"
save_dirname = "dataset/data_shape/groundtrust_image"
if not os.path.isdir(save_dirname):
    os.mkdir(save_dirname)
for i in os.listdir(annot_dirname):
    shape_dicts = get_poly(annot_dirname+'/'+i)
    create_multi_masks(annot_dirname+'/'+i, shape_dicts, i)