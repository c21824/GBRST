import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import random

import os
import cv2

from tqdm import tqdm
import albumentations as A
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import timm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchsummary import summary

from dataset import CustomDataset
from model import CombinedModel

import torch
from torchvision import transforms
from PIL import Image



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CombinedModel().to(device)
model.load_state_dict(torch.load('best_model_weights.pt', map_location=device))
summary(model, input_size=(3, 112, 112))

# dictionary to label all traffic signs class.
classes = {
    0: 'Speed limit (20km/h)', 1: 'Speed limit (30km/h)',
    2: 'Speed limit (50km/h)', 3: 'Speed limit (60km/h)',
    4: 'Speed limit (70km/h)', 5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)', 7: 'Speed limit (100km/h)',
    8: 'Speed limit (120km/h)', 9: 'No passing',
    10: 'No passing veh over 3.5 tons', 11: 'Right-of-way at intersection',
    12: 'Priority road', 13: 'Yield',
    14: 'Stop', 15: 'No vehicles',
    16: 'Veh > 3.5 tons prohibited', 17: 'No entry',
    18: 'General caution', 19: 'Dangerous curve left',
    20: 'Dangerous curve right', 21: 'Double curve',
    22: 'Bumpy road', 23: 'Slippery road',
    24: 'Road narrows on the right', 25: 'Road work',
    26: 'Traffic signals', 27: 'Pedestrians',
    28: 'Children crossing', 29: 'Bicycles crossing',
    30: 'Beware of ice/snow', 31: 'Wild animals crossing',
    32: 'End speed + passing limits', 33: 'Turn right ahead',
    34: 'Turn left ahead', 35: 'Ahead only',
    36: 'Go straight or right', 37: 'Go straight or left',
    38: 'Keep right', 39: 'Keep left',
    40: 'Roundabout mandatory', 41: 'End of no passing',
    42: 'End no passing veh > 3.5 tons'
}
# initialise GUI
top = tk.Tk()
top.geometry('800x600')
top.title('Nhận dạng biển báo giao thông ')
top.configure(background='#ffffff')

label = Label(top, background='#ffffff', font=('arial', 15, 'bold'))
sign_image = Label(top)

def classify(file_path):
    global label_packed
    global model, classes

    # Kiểm tra nếu model đang trên GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Mở và resize ảnh
    image = Image.open(file_path).convert('RGB')
    image = image.resize((112, 112))

    # Chuyển đổi ảnh từ PIL.Image sang Tensor
    transform = transforms.ToTensor()
    image_tensor = transform(image).unsqueeze(0)

    # Chuyển tensor của ảnh sang cùng thiết bị với mô hình
    image_tensor = image_tensor.to(device)

    # predict classes
    with torch.no_grad():
        pred_probabilities = model(image_tensor)[0]
        pred = pred_probabilities.argmax(axis=-1).item()

    # Lấy tên class dựa trên dự đoán
    sign = classes[pred]

    # In và hiển thị kết quả
    print(sign)
    label.configure(foreground='#011638', text=sign)



def show_classify_button(file_path):
    classify_b = Button(top, text="Nhận dạng", command=lambda: classify(file_path), padx=10, pady=5)
    classify_b.configure(background='#c71b20', foreground='white', font=('arial', 10, 'bold'))
    classify_b.place(relx=0.79, rely=0.46)


def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)

        # Tăng kích thước ảnh kết quả (ví dụ: 300x300)
        uploaded = uploaded.resize((300, 300))  # Chỉnh lại kích thước theo nhu cầu

        im = ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image = im  # Lưu tham chiếu để tránh bị xóa bởi garbage collector
        label.configure(text='')
        show_classify_button(file_path)
    except Exception as e:
        print(f"Error: {e}")
        pass


upload = Button(top, text="Upload an image", command=upload_image, padx=10, pady=5)
upload.configure(background='#c71b20', foreground='white', font=('arial', 10, 'bold'))

upload.pack(side=BOTTOM, pady=50)
sign_image.pack(side=BOTTOM, expand=True)
label.pack(side=BOTTOM, expand=True)
heading = Label(top, text="Nhận dạng biển báo giao thông", pady=10, font=('arial', 20, 'bold'))
heading.configure(background='#ffffff', foreground='#364156')

heading1 = Label(top, text="Môn Học: lập trình Python", pady=10, font=('arial', 20, 'bold'))
heading1.configure(background='#ffffff', foreground='#364156')

heading.pack()
heading1.pack()
top.mainloop()