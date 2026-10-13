import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from fastapi import FastAPI, UploadFile
from fastapi.staticfiles import StaticFiles
import uvicorn

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

device = torch.device("cpu")
model = models.resnet18()
in_feature = model.fc.in_features
model.fc = torch.nn.Linear(in_feature, 2)
model.load_state_dict(torch.load("cat_dog_model.pth", map_location=device))
model.eval()

trans = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])

@app.post("/predict")
async def predict(file: UploadFile):
    img = Image.open(file.file).convert("RGB")
    img_tensor = trans(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(img_tensor)
        idx = torch.argmax(out, dim=1).item()
    cls_name = "猫" if idx ==0 else "狗"
    return {"result": cls_name}

if __name__ == "__main__":
    uvicorn.run("predict_api:app", host="127.0.0.1", port=8000)


