import csv
import glob
import os
import random
import torch
import re
import pandas as pd
from torch import optim, nn
from torch.utils.data import DataLoader
from MyDataset import MyDataset
from tqdm import tqdm
from net.Lenet5 import Lenet5
from net.resnet import ResNet18
from net.trained_resnet import trained_model
from net.TwoStream import TwoStreamResNet
from net.resnetx import ModifiedResNet18
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
# from focal_loss import SparseCategoricalFocalLoss
# from focal_loss import FocalLoss
best_acc = 0
best_epoch = 0
batch_size = 24
epochs = 10
extend_epochs = 20
num_classes = 18
data_path = "./dataset/"
matrix2_path = "error_matrix2_pre.txt"
pretrained = True
learning_rate = 0.001

def save_to_txt(error_matrix, file_path):
    if not os.path.exists(file_path):
        with open(file_path, "w") as file:
            for row in error_matrix:
                file.write(" ".join(map(str, row)) + "\n")
    else:
        old_error_matrix = []
        with open(file_path, "r") as file:
            for line in file:
                row = list(map(int, line.strip().split()))
                old_error_matrix.append(row)
        result = []
        for i in range(len(old_error_matrix)):
            row_result = []
            for j in range(len(old_error_matrix[0])):
                sum_element = old_error_matrix[i][j] + error_matrix[i][j]
                row_result.append(sum_element)
            result.append(row_result)
        # print(error_matrix)
        with open(file_path, "w") as file:
            for row in result:
                file.write(" ".join(map(str, row)) + "\n")

def dataset(root):
    name2label = {}  # {"name": label}
    for name in os.listdir(os.path.join(root)):
        if not os.path.isdir(os.path.join(root, name)):  # is dir ?
            continue
        pattern = r'^(\d{1,2})'
        match = re.match(pattern, name)
        name2label[name] = int(match.group(1))-1
    train = []
    val = []
    test = []

    for name in name2label.keys():
        # 获取每个子文件夹中的图片
        folder_images = []
        folder_images += glob.glob(os.path.join(root, name, "*.jpg"))
        folder_images += glob.glob(os.path.join(root, name, "*.png"))
        folder_images += glob.glob(os.path.join(root, name, "*.jpeg"))
        # 保留倒数第二个字符串为"1"的文件名

        folder_images = [file_name for file_name in folder_images if file_name.split("_")[-2] == "0"]
        #folder_images = [file_name for file_name in folder_images if data_dict.get(file_name, 0) < 20]
        # 按照8:1:1的比例划分数据集
        random.shuffle(folder_images)
        train_images = folder_images[:int(0.6 * len(folder_images))]
        val_images = folder_images[int(0.6 * len(folder_images)):int(0.8 * len(folder_images))]
        test_images = folder_images[int(0.8 * len(folder_images)):]
        train += train_images
        val += val_images
        test += test_images
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)
    with open(os.path.join("train_set.csv"), mode="w", newline='') as f:
        writer = csv.writer(f)
        for img in train:
            name = img.split(os.sep)[-2]
            label = name2label[name]
            writer.writerow([img, label])
    with open(os.path.join("val_set.csv"), mode="w", newline='') as f:
        writer = csv.writer(f)
        for img in val:
            name = img.split(os.sep)[-2]
            label = name2label[name]
            writer.writerow([img, label])
    with open(os.path.join("test_set.csv"), mode="w", newline='') as f:
        writer = csv.writer(f)
        for img in test:
            name = img.split(os.sep)[-2]
            label = name2label[name]
            writer.writerow([img, label])

dataset(data_path)

train_dataset = MyDataset("train_set.csv", 224)
train_loader = DataLoader(train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=4)
val_dataset = MyDataset("val_set.csv", 224)
val_loader = DataLoader(val_dataset,
                        batch_size=batch_size,
                        num_workers=4)
test_dataset = MyDataset("test_set.csv", 224)
test_loader = DataLoader(test_dataset,
                         batch_size=batch_size,
                         num_workers=4)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model= TwoStreamResNet(num_classes).to(device)
model= ModifiedResNet18(num_classes).to(device)
criterion = nn.CrossEntropyLoss().to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

error_matrix = [[0] * (2) for _ in range(num_classes)]
error_matrix2 = [[0] * (num_classes) for _ in range(num_classes)]

def train(epoch, save_model):
    global best_acc
    global best_epoch
    running_loss = 0.0
    model.train()
    for batch_idx, data in enumerate(train_loader, 1):
        positive_images, negative_images, labels, file_name = data
        positive_images, negative_images, labels = positive_images.to(device), negative_images.to(device), labels.to(
            device)
        outputs = model(positive_images, negative_images)
        optimizer.zero_grad()
        print(outputs, labels)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_idx % 10 == 0:  # 每10个batch输出一次损失
            print("[%d, %d] loss: %.3f" % (epoch, batch_idx, running_loss % 100))
            running_loss = 0.0
    if epoch % 1 == 0:
        val_acc = test(model, val_loader)
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            torch.save(model.state_dict(), save_model)

    print("Accuracy Of Validation Set:", val_acc * 100.0, "%")
    with open("speed.txt", 'a') as file:
        file.write(str(val_acc)+ "\n")
    print("Best Epoch:", best_epoch, ", Accuracy Of Validation Set:", best_acc * 100, "%")

    # if epoch == epochs:
    #     model.load_state_dict(torch.load(save_model))
    #     test_acc = test2(model, test_loader)
    #     print("Accuracy Of Test Set:", test_acc * 100.0, "%")


def test(model, loader):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in loader:
            positive_images, negative_images, labels, file_name = data
            positive_images, negative_images, labels = positive_images.to(device), negative_images.to(device), labels.to(device)
            outputs = model(positive_images, negative_images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        # print("Accuracy on test set : %.3f%%" % (100 * correct / total), " [%d / %d]" % (correct, total))
        return round(correct / total, 3)

def test2(model, loader):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in loader:
            positive_images, negative_images, labels, file_name = data
            positive_images, negative_images, labels = positive_images.to(device), negative_images.to(device), labels.to(device)
            outputs = model(positive_images, negative_images)
            _, predicted = torch.max(outputs.data, dim=1)

            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            wrong_indices = (predicted != labels).nonzero(as_tuple=True)[0]
            right_indices = (predicted == labels).nonzero(as_tuple=True)[0]
            # if epoch>20:
            for idx in wrong_indices:
                actual_label = labels[idx].item()
                predicted_label = predicted[idx].item()
                # 根据预测结果和真实标签确定错误的类别
                error_matrix[actual_label][0] += 1
                error_matrix[actual_label][1] += 1
                error_matrix2[actual_label][predicted_label] += 1
            for idx in right_indices:
                actual_label = labels[idx].item()
                predicted_label = predicted[idx].item()
                # 根据预测结果和真实标签确定错误的类别
                error_matrix[actual_label][1] += 1
                error_matrix2[actual_label][predicted_label] += 1
        # print("Accuracy on test set : %.3f%%" % (100 * correct / total), " [%d / %d]" % (correct, total))
        return round(correct / total, 3)

# if __name__ == '__main__':
#     for epoch in range(1, epochs + 1):
#         train(epoch, "model/modifiedmodel.mdl")
#     for i in range(num_classes):
#         error_matrix[i][2]=(error_matrix[i][1]-error_matrix[i][0])/error_matrix[i][1]
#         print(error_matrix[i][0],error_matrix[i][1],error_matrix[i][2])
#     for i in error_matrix2:
#         print(i)

for param in model.upper_resnet.parameters():
    param.requires_grad = False
for param in model.lower_resnet.parameters():
    param.requires_grad = False
trainable_params = list(model.CrossResidualBlock.parameters())
trainable_params += list(model.fc1.parameters())
trainable_params += list(model.fc2.parameters())
# trainable_params += list(model.fc3.parameters())

# trainable_params += list(model.fc4.parameters())

trainable_params += list(model.fc5.parameters())
trainable_params += list(model.fc6.parameters())
trainable_params += list(model.fc7.parameters())
trainable_params += list(model.fc8.parameters())
trainable_params += list(model.fc9.parameters())



optimizer = torch.optim.Adam(trainable_params, lr=0.001)
for epoch in range(1, epochs + 1):
    train(epoch, "model/modifiedmodelplus.mdl")

for param in model.parameters():
    param.requires_grad = True
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
for epoch in range(1, extend_epochs+1):
    train(epoch, "model/modifiedmodelplus.mdl")

model.load_state_dict(torch.load("model/modifiedmodelplus.mdl"))

test_acc = test2(model, test_loader)
print("Accuracy Of Test Set:", test_acc * 100.0, "%")

with open("error_matrix1.txt", "a") as file1:
    for row in error_matrix:
        file1.write(" ".join(map(str, row)) + "\n")

with open("error_matrixep.txt", "a") as file2:
    for row in error_matrix2:
        file2.write(" ".join(map(str, row)) + "\n")
    file2.write("\n")
# save_to_txt(error_matrix, "error_matrix.txt")
save_to_txt(error_matrix2, matrix2_path)

