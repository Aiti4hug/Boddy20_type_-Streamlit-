from fastapi import FastAPI, UploadFile, File, HTTPException
import io
import uvicorn
import torch
from torchvision import transforms
import torch.nn as nn
from PIL import Image
import streamlit as st
import os

#gray
transform_data_g = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Grayscale(),
    transforms.ToTensor()
])


#rgb
transform_data_r = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])


#gray
class BoddyG(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.second = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 64 * 64, 128),
            nn.ReLU(),
            nn.Linear(128, 20),
        )

    def forward(self, x):
        x = self.first(x)
        x = self.second(x)
        return x


#rgb
class BoddyR(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.second = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 64 * 64, 128),
            nn.ReLU(),
            nn.Linear(128, 20),
        )

    def forward(self, x):
        x = self.first(x)
        x = self.second(x)
        return x

check_image_app = FastAPI()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

bod_gray = BoddyG().to(device)
bod_rgb = BoddyR().to(device)

bod_gray.load_state_dict(torch.load('mod/boddy_g.pth', map_location=device))
bod_gray.eval()

bod_rgb.load_state_dict(torch.load('mod/boddy_r.pth', map_location=device))
bod_rgb.eval()


classes = [
    'Audi',
    'Bmw',
    'Burger',
    'Cottage_Cabin',
    'Mansion_Luxury_Villa',
    'Mercedes',
    'Nissan',
    'Pasta',
    'Pizza',
    'Salad',
    'Shawarma',
    'Toyota',
    'apartment_flat',
    'ball',
    'hotel',
    'knight',
    'minecraft',
    'moon',
    'single-family_house',
    'sun'
]

st.title('BODDY CLASSIFIER MODEL')
st.text('LOAD image')
st.image('Mitaa.jpg')

type_image = st.file_uploader('drop images', type=['png', 'jpg', 'jpeg'])

if type_image is not None:
    st.image(type_image, caption='loader')

model_type = st.selectbox(
    "Choose model type",
    ["GRAY model", "RGB model"]
)

if st.button('Determine the number'):
    try:
        if type_image is None:
            st.error("Please upload an image first!")
            st.stop()

        image = Image.open(type_image).convert("RGB")

        if model_type == "GRAY model":
            tensor = transform_data_g(image).unsqueeze(0).to(device)
            model = bod_gray
        else:
            tensor = transform_data_r(image).unsqueeze(0).to(device)
            model = bod_rgb

        with torch.no_grad():
            pred = model(tensor).argmax(1).item()

        st.success(f"Answer: {classes[pred]}")

    except Exception as e:
        st.error(f"Error: {str(e)}")
