from fastapi import FastAPI, UploadFile, File
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import io
from torchvision import transforms

# === helper_lib ===
from helper_lib.data_loader import get_data_loader
from helper_lib.model import get_model
from helper_lib.trainer import train_model, train_gan
from helper_lib.generator import generate_samples

# ------------------------------
# FastAPI Initialization
# ------------------------------
app = FastAPI(title="CNN + GAN API", version="2.0")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------
# CNN model part
# ------------------------------
print("🔹 Initializing CNN Model...")
cnn_model = get_model("CNN").to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)

try:
    cnn_model.load_state_dict(torch.load("cnn_model.pth", map_location=device))
    print("✅ Loaded existing CNN model weights.")
except:
    print("Training new CNN model...")
    train_loader = get_data_loader("./data", batch_size=64, train=True)
    cnn_model = train_model(cnn_model, train_loader, criterion, optimizer, device=device, epochs=5)
    torch.save(cnn_model.state_dict(), "cnn_model.pth")
    print("✅ CNN Model saved to cnn_model.pth")


# ------------------------------
# GAN model part
# ------------------------------
print("🔹 Initializing GAN Model...")
gan_models = get_model("GAN")   # {'generator': G, 'discriminator': D}
G, D = gan_models["generator"].to(device), gan_models["discriminator"].to(device)


# ------------------------------
# 路由 1: 首页
# ------------------------------
@app.get("/")
def home():
    return {"message": "✅ CNN + GAN API is running!"}


# ------------------------------
# 路由 2: CNN 预测接口
# ------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = cnn_model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        label = predicted.item()

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    return {"predicted_class": label, "label": class_names[label]}


# ------------------------------
# 路由 3: GAN 训练接口
# ------------------------------
from torchvision import datasets, transforms

@app.get("/train_gan")
def train_gan_endpoint(epochs: int = 3):
    print("🚀 Training GAN model...")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # 单通道MNIST标准化
    ])
    mnist_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(mnist_data, batch_size=64, shuffle=True)

    trained_G, trained_D = train_gan(gan_models, train_loader, device=device, epochs=epochs)
    torch.save(trained_G.state_dict(), "gan_generator.pth")
    torch.save(trained_D.state_dict(), "gan_discriminator.pth")
    return {"message": f"✅ GAN trained for {epochs} epochs and saved to disk."}


# ------------------------------
# 路由 4: GAN 生成接口
# ------------------------------
from fastapi.responses import FileResponse

@app.get("/generate_gan")
def generate_gan_endpoint(num_samples: int = 5):
    print(f"🖼️ Generating {num_samples} GAN images...")
    try:
        G.load_state_dict(torch.load("gan_generator.pth", map_location=device))
        print("✅ Loaded trained GAN generator weights.")
    except:
        print("⚠️ No pretrained GAN found. Using untrained generator.")
    output_path = generate_samples(G, device=device, num_samples=num_samples)
    return {"message": f"✅ Generated {num_samples} images", "file_path": output_path}