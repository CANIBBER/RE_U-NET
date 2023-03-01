
import torch
import albumentations as A
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from re_NET_lw import REUNET
from albumentations.pytorch.transforms import ToTensorV2
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import pylab
import numpy as np

from google.colab.patches import cv2_imshow



from utils import(
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

TRAIN_IMG_DIR = "Carvanadata/train_images/"
TRAIN_MASK_DIR = "Carvanadata/train_masks/"
VAL_IMG_DIR = "Carvanadata/train_images/"
VAL_MASK_DIR = "Carvanadata/train_masks/"
LOAD_MODEL =True
BATCH_SIZE = 1
NUM_WORKERS = 1
IMAGE_HEIGHT = 480
IMAGE_WIDTH = 480
PIN_MENMORY = True
LEARNING_RATE = 1e-4
train_transform = A.Compose(
    [
        A.Resize(height = IMAGE_HEIGHT, width = IMAGE_WIDTH),
        A.Rotate(limit = 35, p =1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Normalize(
            mean = [0.0, 0.0, 0.0],
            std = [1.0, 1.0, 1.0],
            max_pixel_value = 225.0
        ),
        ToTensorV2(),
    ],
)

val_transform = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=225.0
        ),
        ToTensorV2(),
    ]
)
train_loader,val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transform
    )

def main():
    
    DEVICE = 'cuda' if torch.cuda.is_available() else "cpu"
    model = REUNET(in_channels=3, out_channels=1).to(DEVICE)
    
    

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"),model)
        check_accuracy(val_loader, model, device =DEVICE)
    N = 0
    for image,label in val_loader:
        inpu = image.to(device = DEVICE)
        
        prediction = model(inpu)
        prediction = torch.sigmoid(prediction)
        prediction = (prediction>0.5).float()
        plt.figure()
        plt.subplot(1,3,1)
        image = image.squeeze(0)
        image = (image)

        plt.imshow(image.T)
        plt.title('origin')
        
        plt.subplot(1,3,2)
        prediction = prediction.squeeze(0).cpu()
        prediction = prediction.detach().numpy()
        

        plt.imshow(prediction.squeeze().T, cmap=plt.cm.gray)       
        plt.title('prediction')
        
        plt.subplot(1,3,3)
        plt.imshow(label.squeeze().T, cmap=plt.cm.gray)       
        plt.title('label')
        
        # plt.show()   
        # pylab.show()
        # cv2_imshow(plt)
        plt.savefig('/content/drive/MyDrive/K-MEANS/UNET/DEMO/demo_'+str(N)+'_O.jpg')
        N+=1
        
        cv2.waitKey(25)
        
if  __name__ =="__main__":
    main()
                            
                
                
            