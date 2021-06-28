# Multiclass-Semantic-Segmentation-UNet_series-pytorch
Semantic-Segmentation-Multiclass_FCN_Unet_ResUnet(Multiclass)  
run train.py 執行train.py即可訓練  
run test.py   執行test.py即可使用所訓練的pt檔預測輸出，輸出目錄為dataset/result，檔名會新增_predict以辨識  
run eval.py   執行eval.py即可評估dataset/result內的輸出圖像跟ground trust圖像，將會在目錄輸出Model_Predict.txt，內含Average Pixel Accuracy | classPixelAccuracy | Average Mean Accuracy | Average Mean IU    
  
## **Implementation Details**  
Python 3.6  
Pytorch 1.8  
Cuda 10.2  
input images are resized to [256 ,256]   
Training was performed using Stochastic Gradient Descent (SGD) with learning rate 0.001    
輸入圖像被縮放至[256 ,256]   使用SGD，學習率0.001  

## **Use Model**  
+ [FCN8S](https://github.com/bat67/pytorch-FCN-easiest-demo)
+ [FCN32S](https://github.com/bat67/pytorch-FCN-easiest-demo)
+ [UNet](https://medium.com/analytics-vidhya/pytorch-implementation-of-semantic-segmentation-for-single-class-from-scratch-81f96643c98c)  
+ [UNet++](https://github.com/4uiiurz1/pytorch-nested-unet)
+ [ResUnet](https://github.com/galprz/brain-tumor-segmentation)  

## **Dataset**  
The dataset is divided into three parts: 160 images for the training set and 40 images for the validation set
The dataset is to segment 4 classes such as circle,square,triangle,star  
資料集被分為160張用來訓練，40張用來驗證，並且有4個類[circle,square,triangle,star]  
Their corresponding grayscale values are
|Class | Grayscale|
|---- | ----|
|circle | 100|
|square |  250|
|triangle |  255|
|star |  200| 
  
## **Result**  　　
|Model | Average Pixel Accurac|Average Mean Accuracy|
|---- | ----|----|
|FCN8S | 0.979|0.807|  

![images](dataset/data_shape/vaild_annot_mask/162.json_mask.png)  ![images](dataset/result/162.json_mask.png_predict.png)  
　　　　　Ground Trust 　　　　　　　　　　Predict Image

