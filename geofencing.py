import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
from torch_snippets import *
import cv2 as cv
import numpy as np
from torchvision.ops import nms


label2target={'vehicle': 1, 'background': 0}
target2label={1: 'vehicle', 0: 'background'}

num_classes=2
def get_model():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model



def decode_output(output):
    'convert tensors to numpy arrays'
    bbs = output['boxes'].cpu().detach().numpy().astype(np.uint16)
    labels = np.array([target2label[i] for i in output['labels'].cpu().detach().numpy()])
    confs = output['scores'].cpu().detach().numpy()
    ixs = nms(torch.tensor(bbs.astype(np.float32)), torch.tensor(confs), 0.08)
    bbs, confs, labels = [tensor[ixs] for tensor in [bbs, confs, labels]]

    if len(ixs) == 1:
        bbs, confs, labels = [np.array([tensor]) for tensor in [bbs, confs, labels]]
    return bbs.tolist(), confs.tolist(), labels.tolist()


drawing=False
ix,iy,sx,sy=-1,-1,-1,-1
coordinates=[]
box_coordinates=[]
poly=[]
def draw_rectangle(event,x,y,flags,param):
    global ix,iy,sx,sy
    if event==cv.EVENT_LBUTTONDOWN:
        cv.circle(frame,(x,y),3,(0,0,127),-1)
        

        if ix!=-1:
            cv.line(frame,(ix,iy),(x,y),(0,0,127),2,cv.LINE_AA)
            coordinates.append([ix,iy,x,y])
            
        else:
            sx,sy=x,y
        ix,iy=x,y
    

    elif cv.waitKey(1) & 0xFF == ord('q') :
        ix, iy = -1, -1 # reset ix and iy
        box_coordinates.append(coordinates.copy())
        coordinates.clear()
    
    


mask=np.zeros((360,640))

model=get_model()
model.load_state_dict(torch.load('/Users/ishvaksud/Downloads/a.pth',map_location=torch.device('cpu')))
model.eval()

count=0
cap=cv.VideoCapture('/Users/ishvaksud/Desktop/carPark.mp4')
mask = np.zeros((712,712))
middle=[]
while True:
    ret,frame=cap.read()
    frame=cv.resize(frame,(640,640))
    if count==0:
        
        cv.namedWindow('geo')
        cv.setMouseCallback('geo',draw_rectangle)
        while(1):
            cv.imshow('geo',frame)
            if cv.waitKey(20) & 0xFF == 27:
                count=count+1
                break
        
    mask = np.zeros_like(frame)
    
   
    for box in box_coordinates:
        for i in box:
            cv.line(frame,(i[0],i[1]),(i[2],i[3]),(0,255,255),2)

    for polygon in box_coordinates:
        pts = np.array(polygon, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], (255, 255, 255))
    

    
        
    roi = cv.bitwise_and(frame, mask)
    roi=cv.resize(roi,(640,640))
    img=torch.tensor(roi).float().permute(2,0,1)
    img=img/255.
    img=img.unsqueeze(0)
    outputs=model(img)
    for ix, output in enumerate(outputs):
        bbs, confs, labels = decode_output(output)
    for i in bbs:
        xmin,ymin,xmax,ymax=i[0],i[1],i[2],i[3]
        cv.rectangle(frame,(xmin,ymin),(xmax,ymax),(255,0,0),2)
    cv.imshow('detection',frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

    
cv.destroyAllWindows() 
cap.release()


    




        