import numpy as np
import torchvision
import torch
import torchvision.transforms as transforms
from PIL import Image,ImageDraw,ImageFont
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from torch.utils import data
import glob
#model=torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
#print(model.eval())
#torch.save(model,"./model.pkl")
model=torch.load("model.pkl")
print(model.eval())
#image = Image.open("F:/COCO2017/test2017/000000001124.jpg")
#test_data_dir = "F:/object detection dataset/"
image_list = []
for filename in glob.glob("F:/object detection dataset/all/*.jpg"):
    im = Image.open(filename)
    image_list.append(im)
#image = Image.open("F:/object detection dataset/bicycle/000000014311.JPG")
transform_d = transforms.Compose([transforms.ToTensor()])
i=0
for image in image_list:
    #image = image_list[i]
    image_t = transform_d(image)
    pred = model([image_t])
    print(pred)
    COCO_CATEGORY_NAMES = ['__background__', 'person', 'bicycle', 'car', 'motorbike', 'aeroplane',
                           'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A ', 'stop sign',
                           'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                           'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
                           'handbag',
                           'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
                           'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                           'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                           'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
                           'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'N/A', 'diningtable', 'N/A', 'N/A', 'toilet',
                           'N/A',
                           'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microphone',
                           'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors',
                           'teddy bear', 'hair drier', 'toothbrush']

    pred_class = [COCO_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())]
    pred_score = list(pred[0]['scores'].detach().numpy())
    pred_boxes = [[i[0], i[1], i[2], i[3]] for i in list(pred[0]['boxes'].detach().numpy())]
    pred_index = [pred_score.index(x) for x in pred_score if x > 0.9]
    #pred_index = [pred_score.index(pred_score[0])]
    fontsize = np.int16(image.size[1] / 30)
    font1 = ImageFont.truetype("C:/Windows/Fonts/STXIHEI.ttf",fontsize)
    draw = ImageDraw.Draw(image)
    for index in pred_index:
        box = pred_boxes[index]
        draw.rectangle(box, outline='red')
        texts = pred_class[index] + ":" + str(np.round(pred_score[index], 2))
        draw.text((box[0], box[1]), texts, fill="yellow",font=font1)
    #print(image.show())
    i+=1
    image.save("F:/object detection dataset/detection/{}.jpg".format(i))
