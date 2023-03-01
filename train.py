import torch
import albumentations as A
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from re_NET_lw import REUNET
from albumentations.pytorch.transforms import ToTensorV2

from utils import(
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)
#Hyperparameters etc
LEARNING_RATE = 1e-3
DEVICE = 'cuda' if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
NUM_WORKERS = 1
IMAGE_HEIGHT = 480
IMAGE_WIDTH = 480
PIN_MENMORY = True
LOAD_MODEL =False
TRAIN_IMG_DIR = "Carvanadata/train_images/"
TRAIN_MASK_DIR = "Carvanadata/train_masks/"
VAL_IMG_DIR = "Carvanadata/train_images/"
VAL_MASK_DIR = "Carvanadata/train_masks/"

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    alloss = 0
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device = DEVICE)
        targets = targets.float().to(device = DEVICE)

        with torch.cuda.amp.autocast():
            predictions = model(data)
            predictions = torch.squeeze(predictions, 0)
            loss = loss_fn(predictions, targets)
        alloss += loss
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


        loop.set_postfix(loss = loss.item())
    counting_loss = float(alloss)
    print("Counting_loss: %s" % (counting_loss))
    print("---------------")

def main():
# transform用于将图像输入标准化，归一化，实现一定程度的图片旋转避免过收敛
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

    model = REUNET(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transform
    )
    #加载模型
    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)
        check_accuracy(val_loader, model, device =DEVICE)

    scaler = torch.cuda.amp.GradScaler()
    #指定训练世代
    NUM_EPOCHS = 3
    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

       # save model
        checkpoint ={
            "state_dict":model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        save_checkpoint(checkpoint["state_dict"], filename = "my_checkpoint.pth.tar")

       # check accuracy
        check_accuracy(val_loader, model, device =DEVICE)
       # print sample
        # save_predictions_as_imgs(
        #     val_loader, model, DEVICE, folder ="saved_images/"
        # )
if __name__ =="__main__":
    main()