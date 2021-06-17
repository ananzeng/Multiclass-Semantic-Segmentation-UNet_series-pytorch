# Semantic-Segmentation_FCN_Unet_ResUnet
Semantic-Segmentation-Multiclass_FCN_Unet_ResUnet(Multiclass)  
run train.py 執行train.py即可訓練  
run test.py   執行test.py即可使用所訓練的pt檔預測輸出，輸出目錄為dataset/result，檔名會新增_predict以辨識  
run eval.py   執行eval.py即可評估dataset/result內的輸出圖像跟ground trust圖像，將會在目錄輸出Model_Predict.txt，內含Average Pixel Accuracy | classPixelAccuracy | Average Mean Accuracy | Average Mean IU    
  
Implementation Details  
input images are resized to [256 ,256]   
Training was performed using Stochastic Gradient Descent (SGD) with learning rate 0.001    

The dataset is divided into three parts: 160 images for the training set and 40 images for the validation set
The dataset is to segment 4 classes such as circle,square,triangle,star
