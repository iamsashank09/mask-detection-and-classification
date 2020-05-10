# Mask Detection and Classification | Mask or No Mask detection | Deep Learning and Computer Vision

A deep understanding of the current COVID-19 pandemic shows how one person's negligence can cause widespread harm. Since the general public is getting back on the streets and employees going back to their workplaces, it is extremely important for everyone to adorn face-masks as suggested by WHO, to keep themselves and others safe. [WHO: value of masks](https://www.who.int/emergencies/diseases/novel-coronavirus-2019/advice-for-public/when-and-how-to-use-masks)

The application detects persons faces and classifies if they are wearing a mask, or not. It shows a bounding box in the video feed and classifies the faces in the frame. 

### Here is a demo containing the application output: 
![enter image description here](https://github.com/iamsashank09/mask-detection-and-classification/blob/master/outputs/GIFoutput_video.gif)

![enter image description here](https://github.com/iamsashank09/mask-detection-and-classification/blob/master/outputs/output_P1.gif)

![enter image description here](https://github.com/iamsashank09/mask-detection-and-classification/blob/master/outputs/output_P2.gif)


### How to use the application:

The maskDetectionApp.py file contains the code to run the application. Before which, you need to download the trained model from [HERE](https://drive.google.com/open?id=19TwyycoDwUVJ3DWdsHJGbOXIgp8nvn2e) and place it in the folder "models". Example: "models/Res18oneFC_model.pth"

##### Running it on live webcam feed:

    python3 maskDetectionApp.py

##### Running it on a video:

    python3 maskDetectionApp.py video

Then the program will ask for the path of the video, please pass the path to the video file and the output will be saved in the "outputs" folder.

### Concerns over such monitoring applications:
As with my previous project on [Social Distance monitoring application](https://github.com/iamsashank09/social-distance-dashboard).
While working on this project, I started having concerns over how this kind of monitoring can be used to create spaces with no freedom, which is scary. I understand this kind of work can be used to boost authoritarianism and force suppression. I can only hope we always keep in mind the freedoms of the individual while we build a safe society.  


### Resources:
The [Medical Mask Dataset](https://www.kaggle.com/vtech6/medical-masks-dataset) from Kaggle.

Input video used for testing and development: [CBS17 Youtube](https://www.youtube.com/watch?v=VCR6lzXPy2A)

