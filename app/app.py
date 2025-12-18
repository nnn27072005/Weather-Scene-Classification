import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import joblib
# ==========================================
# PHẦN 1: ĐỊNH NGHĨA MODEL (BẮT BUỘC)
# ==========================================
# Bạn HÃY COPY PASTE toàn bộ Class ResNet, DenseNet, 
# ResidualBlock, DenseBlock vào đây để torch.load không bị lỗi.

# Ví dụ (Chỉ là placeholder, bạn thay bằng code thật của bạn):
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride), nn.BatchNorm2d(out_channels))
        self.relu = nn.ReLU()
    def forward(self, x):
        shortcut = x.clone()
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x += self.downsample(shortcut)
        x = self.relu(x)
        return x
            

class ResNet(nn.Module):
    def __init__(self, residual_block, n_blocks_lst, n_classes):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64 , kernel_size = 7, stride = 2, padding=3)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = self.create_layer(residual_block, 64, 64, n_blocks_lst[0], 1)
        self.conv3 = self.create_layer(residual_block, 64, 128, n_blocks_lst[1], 2)
        self.conv4 = self.create_layer(residual_block, 128, 256, n_blocks_lst[2], 2)
        self.conv5 = self.create_layer(residual_block, 256, 512, n_blocks_lst[3], 2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512, n_classes)
    def create_layer(self, residual_block, in_channels, out_channels, n_blocks, stride):
        blocks = []
        first_block = residual_block(in_channels, out_channels, stride)
        blocks.append(first_block)
        for idx in range(1, n_blocks):
            block = residual_block(out_channels, out_channels, stride = 1)
            blocks.append(block)
        block_sequential = nn.Sequential(*blocks)

        return block_sequential

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.maxpool(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return x

class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(BottleneckBlock,self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, 4*growth_rate,
                               kernel_size =1, bias = False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate,
                              kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        res = x
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = torch.cat([res,x], 1)
        return x

class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate):
        super(DenseBlock,self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(BottleneckBlock(in_channels + i*growth_rate
                         , growth_rate))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class DenseNet(nn.Module):
    def __init__(self, num_blocks, growth_rate, num_classes):
        super(DenseNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 2*growth_rate, kernel_size=7, 
                              padding=3, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(2*growth_rate)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.dense_blocks = nn.ModuleList()
        in_channels = 2*growth_rate
        
        for i, num_layers in enumerate(num_blocks):
            self.dense_blocks.append(DenseBlock(
                num_layers, in_channels, growth_rate
            ))
            in_channels += num_layers * growth_rate
            if i != len(num_blocks) -1:
                out_channels = in_channels // 2
                self.dense_blocks.append(nn.Sequential(
                    nn.BatchNorm2d(in_channels),
                    # Thêm ReLU inplace vào transition layer để tiết kiệm RAM
                    nn.ReLU(inplace=True), 
                    nn.Conv2d(in_channels, out_channels, 
                    kernel_size=1, bias=False),
                    nn.AvgPool2d(kernel_size=2, stride=2) # Transition thường có pooling
                ))
                in_channels = out_channels

        self.bn2 = nn.BatchNorm2d(in_channels)
        
        # --- SỬA LỖI SHAPE TẠI ĐÂY ---
        # Thay vì AvgPool2d(7), dùng AdaptiveAvgPool2d((1, 1))
        # Nó ép mọi kích thước ảnh về 1x1 -> Flatten sẽ luôn khớp với self.fc
        self.pool2 = nn.AdaptiveAvgPool2d((1, 1)) 
        
        # --- SỬA LỖI MEMORY TẠI ĐÂY ---
        # Thêm inplace=True để tiết kiệm bộ nhớ GPU
        self.relu = nn.ReLU(inplace=True) 
        
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)
        
        for block in self.dense_blocks:
            x = block(x)
            
        x = self.bn2(x)
        x = self.relu(x)
        
        # Sau bước này, x sẽ có dạng (Batch, Channel, 1, 1)
        x = self.pool2(x) 
        
        # Flatten: (Batch, Channel, 1, 1) -> (Batch, Channel)
        x = torch.flatten(x, 1) 
        
        # Lúc này Channel khớp với in_channels của self.fc -> Hết lỗi
        x = self.fc(x)
        return x

# ==========================================
# PHẦN 2: CẤU HÌNH VÀ LOAD MODEL
# ==========================================

# Định nghĩa các nhãn (Labels) - Bạn hãy sửa lại cho đúng với dataset của bạn
WEATHER_CLASSES = ['dew', 'fogsmog', 'frost', 'glaze', 'hail', 'lightning', 'rain', 'rainbow', 'rime', 'sandstorm', 'snow']
SCENE_CLASSES = ['Buildings', 'Forest', 'Glacier', 'Mountain', 'Sea', 'Street']

def transform(img, img_size=(224, 224)):
    img = img.resize(img_size)
    img = np.array(img)[..., :3] # Lấy 3 kênh màu (bỏ kênh trong suốt nếu có)
    img = torch.tensor(img).permute(2,0,1).float() # Chuyển sang (C, H, W)
    normalized_img = img / 255.0 # Chuẩn hóa về 0-1
    return normalized_img

# Hàm load model (Dùng cache để không phải load lại mỗi lần click)
@st.cache_resource
def load_models():
    device = torch.device('cpu') # Deploy web thường chạy CPU cho rẻ/tiện
    path_weather = '../models/model_weather.pth'
    path_scenes = '../models/model_scenes.pth'
    # Load Model Weather
    try:
        model_weather = joblib.load(path_weather)
        model_weather.eval()
    except Exception as e:
        model_weather = None
        print(f"Lỗi load Weather: {e}")

    # Load Model Scenes
    try:
        model_scenes = joblib.load(path_scenes)
        model_scenes.eval()
    except Exception as e:
        model_scenes = None
        print(f"Lỗi load Scenes: {e}")
        
    return model_weather, model_scenes

model_weather, model_scenes = load_models()

# ==========================================
# PHẦN 3: GIAO DIỆN STREAMLIT
# ==========================================

st.title("📸 AI Image Classifier Hub")
st.sidebar.title("Chọn chức năng")

# Tạo Menu bên trái
app_mode = st.sidebar.selectbox("Chọn loại Model:",
                                ["Dự đoán Thời tiết (Weather)", "Phân loại Khung cảnh (Scenes)"])

if app_mode == "Dự đoán Thời tiết (Weather)":
    st.header("🌤️ Weather Classification")
    model = model_weather
    classes = WEATHER_CLASSES
    
elif app_mode == "Phân loại Khung cảnh (Scenes)":
    st.header("🏞️ Scene Classification")
    model = model_scenes
    classes = SCENE_CLASSES

# Khu vực upload ảnh
uploaded_file = st.file_uploader("Chọn một bức ảnh...", type=["jpg", "png", "jpeg"], key=app_mode)

if uploaded_file is not None:
    # Hiển thị ảnh
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Ảnh đã tải lên', width=400)
    
    if st.button('Dự đoán ngay'):
        if model is None:
            st.error("Chưa tìm thấy file model!")
        else:
            with st.spinner('Đang phân tích...'):
                try:
                    # 1. Gọi hàm transform thủ công của bạn
                    # Lưu ý: Sửa (224, 224) thành kích thước thật bạn đã train (ví dụ (150, 150)?)
                    img_tensor = transform(image, img_size=(224, 224)) 
                    
                    # 2. Quan trọng: Thêm chiều Batch (Batch Dimension)
                    # Từ (3, 224, 224) -> (1, 3, 224, 224)
                    img_tensor = img_tensor.unsqueeze(0) 
                    device = next(model.parameters()).device
                    img_tensor = img_tensor.to(device)
                    # 3. Đưa vào Model
                    with torch.no_grad():
                        # Nếu deploy trên CPU thì không cần .to(device) nếu model đang ở CPU
                        outputs = model(img_tensor)
                        _, predicted = torch.max(outputs, 1)
                        confidence = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
                    
                    # 4. Hiển thị kết quả
                    pred_label = classes[predicted.item()]
                    conf_score = confidence[predicted.item()].item()
                    
                    st.success(f"Kết quả: **{pred_label}**")
                    st.info(f"Độ tin cậy: {conf_score:.2f}%")
                    
                    # Biểu đồ
                    st.bar_chart({classes[i]: confidence[i].item() for i in range(len(classes))})
                    
                except Exception as e:
                    st.error(f"Đã xảy ra lỗi khi xử lý ảnh: {e}")