
import cv2
from mtcnn import MTCNN
import torch 
import torchvision.transforms as transforms
import argparse
from PIL import Image

class MaskDetector:

    def __init__(self, modelPath):
        self.modelPath = modelPath
        self.setup = True

    def maskSetup(self):
        if self.setup:
            checkpoint = torch.load(self.modelPath, map_location='cpu')
            model = checkpoint['model']
            model.load_state_dict(checkpoint['state_dict'])
            for parameter in model.parameters():
                parameter.requires_grad = False
            
            self.model = model.eval()

            self.faceDetector = MTCNN()

            self.transforms = transforms.Compose([
                                       transforms.Resize((224,224)),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
                                       ])
        
            self.setup = False


    def maskProcess(self, frame):
        image = frame.copy()
        displayImg = frame.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.faceDetector.detect_faces(image)
        faces = []
        font_scale=1
        thickness = 2
        font=cv2.FONT_HERSHEY_SIMPLEX

        for i in results:
            faces.append(i['box'])

        for (x, y, w, h) in faces:
                
            x, y = max(0,x), max(0,y)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)

            cropped_img = frame[y:y+h, x:x+w, :]
            temp_image = Image.fromarray(cropped_img, mode = "RGB")
            temp_image = self.transforms(temp_image)
            image = temp_image.unsqueeze(0)

            result = self.model(image)
            _, maximum = torch.max(result.data, 1)
            prediction = maximum.item()

            if prediction == 0:
                cv2.putText(displayImg, "Masked", (x,y - 10), font, font_scale, (0,255,0), thickness)
                cv2.rectangle(displayImg, (x, y), (x+w, y+h), (0,255,0), 2)
            elif prediction == 1:
                cv2.putText(displayImg, "No Mask", (x,y - 10), font, font_scale, (0,0,255), thickness)
                cv2.rectangle(displayImg, (x, y), (x+w, y+h), (0,0,255), 2)

        self.outputFrame = displayImg.copy()

    def maskDisplay(self):
        return self.outputFrame



if __name__ == "__main__":

    # inspired from github/jrosebr1 utils package
    def resize(image, width):
        newShape = None
        (h, w) = image.shape[:2]
        r = width / float(w)
        dim = (width, int(h * r))
        resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        return resized

    parser = argparse.ArgumentParser()
    parser.add_argument('source', nargs='?' , default="live", help="Enter 'video' if you want to run the detector on a video.")
    args = parser.parse_args()

    if args.source.lower() == 'video':
        videoMode = True
        videoPath = input('Enter video path, (example: /media/video/hi.mp4) : ')
        if videoPath[0] == '"':
            videoPath = videoPath[1:-1]
        writer = None
        opname = "outputs/output_" + videoPath.split('/')[-1][:-4] + '.avi'
    else:
        videoMode = False
        videoPath = 0
        writer = "Not required"
        opname = None

    cap = cv2.VideoCapture(videoPath)
    fno = 0

    modelPath = "model/Res50oneFC_model.pth"

    detectorObj = MaskDetector(modelPath)

    while(True):

        ret, frame = cap.read()
        if not ret:
            break


        currentImg = frame.copy()
        currentImg = resize(currentImg, width=480)
        imageShape = currentImg.shape
        fno += 1

        if(fno%1 == 0 or fno == 1):
            detectorObj.maskSetup()
            detectorObj.maskProcess(currentImg)
            outputFrame = detectorObj.maskDisplay()

            if writer is None:
                print("Writing the output file to: ", opname)
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                writer = cv2.VideoWriter(opname, fourcc, 30,
                    (outputFrame.shape[1], outputFrame.shape[0]), True)
        
            cv2.imshow('Mask Detection Dashboard', outputFrame)
        
        if videoMode:
            writer.write(outputFrame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # print("Full time taken {} minutes".format((fulltock-fulltick)/60))
 
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()