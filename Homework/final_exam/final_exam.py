import torch
import torchvision
import torch
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import os
import cv2
import pandas as pd
from utils.pytorchtool import *

batch_size = 16
n_epochs = 200
patience = 20
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406],
                                                    [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

train_data = torchvision.datasets.ImageFolder('/home/ailab/ML/image_data/train',transform=train_transforms)
valid_data = torchvision.datasets.ImageFolder('/home/ailab/ML/image_data/valid',transform=test_transforms)
test_data = torchvision.datasets.ImageFolder('/home/ailab/ML/image_data/test',transform=test_transforms)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
print(len(train_data), len(valid_data),len(train_loader),len(valid_loader))# # functions to show an image
class_names = train_data.class_to_idx
print(class_names)
track=0
for data, label in train_loader:
    track+=+1
    print(track)
    print(label.shape)
    print(data.shape)
print('stop')

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    plt.savefig('show_img.jpg')

def create_ResNet34(learning_rate = 0.015, model_to_retrain = None, use_gpu = True):
    load_resnet_pretrained = True if model_to_retrain is None else False

    model    = torchvision.models.resnet50(pretrained = load_resnet_pretrained)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
                               nn.Linear(num_ftrs, 5)
                            #    nn.LogSoftmax(dim = 1)
                            )

    if model_to_retrain is not None:
        model.load_state_dict(torch.load(model_to_retrain))
        print(f"pre-trained model loaded -> { model_to_retrain }")
        print()

    num_of_gpus = []
    if use_gpu and torch.cuda.is_available():    
        num_of_gpus = [ i for i in range(torch.cuda.device_count()) ]
        print(f"num_of_gpus = { num_of_gpus }")
        if len(num_of_gpus) > 1:
            model = nn.DataParallel(model, device_ids = num_of_gpus)
            # torch.nn.modules.module.ModuleAttributeError: 'ResNet' object has no attribute 'device_ids'
            model.to(f"cuda:{ model.device_ids[0] }")
        else:
            model.cuda()

    # optimizer = torch.optim.RMSprop(model.parameters(), lr = learning_rate)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    loss_func = nn.CrossEntropyLoss()
    #return model, optimizer, loss_func, torch.cuda.is_available()
    return model, optimizer, loss_func, len(num_of_gpus)



model, optimizer, loss_func, num_of_gpus = create_ResNet34()
train_losses = []
valid_losses = []
is_gpu_available = True
early_stopping = EarlyStopping(patience=patience, verbose=True)
for epoch in range(1, n_epochs):
    train_loss = 0.0
    valid_loss = 0.0
    running_corrects = 0.0
    start = time.time()


    for phase in ['training', 'validation']:
        if phase == "training":
            data_loader = train_loader
            model.train()
        else:
            data_loader = valid_loader
            model.eval()
        
        for i, (batch_x, batch_y) in enumerate(data_loader):
            if is_gpu_available:
                batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
            
            if phase == "training":
                optimizer.zero_grad()
            
            predict = model(batch_x)
            loss    = loss_func(predict, batch_y)

            # pred_max = predict.data.max(dim = 1, keepdim = False)[1]
            # acc_cnt  = pred_max.eq(batch_y.data).cpu().sum()

            if phase == "training":
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            else:
                _, preds = torch.max(predict, 1)
                valid_loss += loss.item()
                running_corrects += torch.sum(preds == batch_y.data)
        # end of for i
    # end of phase
    train_loss = train_loss/len(train_loader.dataset)
    valid_loss = valid_loss/len(valid_loader.dataset)
    epoch_acc = running_corrects.double() /len(valid_loader.dataset)
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    print('Epoch: {}-{} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}\tValidation Accuracy: {:.6f}'.format(
        epoch, n_epochs, train_loss, valid_loss, epoch_acc))

    # end of for epoch

    # early_stopping needs the validation loss to check if it has decresed, 
    # and if it has, it will make a checkpoint of the current model
    early_stopping(valid_loss, model)
    
    if early_stopping.early_stop:
        print("Early stopping")
        break

plt.plot(train_losses, label='Training loss')
plt.plot(valid_losses, label='Validation loss')
plt.legend(frameon=False)
plt.savefig("loss_rate.jpg")

model_pt = './resnet_model.pt'
torch.save(model, model_pt)

model = torch.load(model_pt)
model.eval()
predictions = []
with torch.no_grad():
    for i, (batch_x, batch_y) in enumerate(test_loader, 0):
        # print(batch_x.shape)
        predict = model(batch_x)
        _, preds = torch.max(predict, 1)
        fname, _ = test_loader.dataset.samples[i]
        pred = preds.item()
        head, tail = os.path.split(fname)
        # print(tail[:-4], pred)
        predictions.append(pred)


# print(test_data.imgs)

# print(predictions)
def predict_one(image_path, model_path):
    model = torch.load(model_path)
    model.eval()
    head, tail = os.path.split(image_path)
    # print(tail)
    img01 = cv2.imread(image_path)
    resized = cv2.resize(img01, (224,224), interpolation = cv2.INTER_AREA)
    np_images = np.asarray(resized)
    np_images = np_images.reshape(1, np_images.shape[2], np_images.shape[0], np_images.shape[1])
    images_tensor = torch.from_numpy(np_images).type(torch.float)
    predict = model(images_tensor)
    _, pred = torch.max(predict, 1)
    # print(preds.item())
    return tail, pred.item()

data = pd.read_csv('sample_submission.csv')
data['flower_class'] = predictions
data.to_csv('new_submission.csv')
