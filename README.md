# Semantic-Segmentation_FCN_Unet_ResUnet
Semantic-Segmentation-Multiclass_FCN_Unet_ResUnet(Multiclass)  
run train.py 執行train.py即可訓練  
run test.py   執行test.py即可使用所訓練的pt檔預測輸出，輸出目錄為dataset/result，檔名會新增_predict以辨識  
run eval.py   執行eval.py即可評估dataset/result內的輸出圖像跟ground trust圖像，將會在目錄輸出Model_Predict.txt，內含Average Pixel Accuracy | classPixelAccuracy | Average Mean Accuracy | Average Mean IU    
  
## **Implementation Details**  
input images are resized to [256 ,256]   
Training was performed using Stochastic Gradient Descent (SGD) with learning rate 0.001    
輸入圖像被縮放至[256 ,256]   使用SGD，學習率0.001  

## **Use Model**  
+ FCN8S    
+ FCN32S  
+ UNet  
+ UNet++  
+ ResUnet  

## **Dataset**  
The dataset is divided into three parts: 160 images for the training set and 40 images for the validation set
The dataset is to segment 4 classes such as circle,square,triangle,star  
資料及被分為160用來訓練，40張用來驗證，並且有4個類[circle,square,triangle,star]  
Their corresponding grayscale values are
|Class | Grayscale|
|---- | ----|
|circle | 100|
|square |  250|
|triangle |  255|
|star |  200|  

