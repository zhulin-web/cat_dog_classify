
import os
# 【重要】放在最顶部，兜底防止Windows OMP冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm

device = torch.device("cpu")

# 图像预处理
train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])
val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])

# 读取图片+标签
def load_img_label(folder_path, transform):
    data = []
    class_map = {"cats":0, "dogs":1}
    for cls_name, label in class_map.items():
        cls_dir = os.path.join(folder_path, cls_name)
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith(("jpg","png","jpeg")):
                img = Image.open(os.path.join(cls_dir,fname)).convert("RGB")
                img_tensor = transform(img)
                data.append((img_tensor, label))
    return data

train_data = load_img_label("dataset/train", train_transform)
val_data = load_img_label("dataset/val", val_transform)

# 加载预训练ResNet18，二分类
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
in_feature = model.fc.in_features
model.fc = nn.Linear(in_feature, 2)
model.to(device)

loss_fn = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr=1e-4)
epoch_num = 10

for epoch in range(epoch_num):
    model.train()
    total_loss = 0
    for img_tensor, label in tqdm(train_data):
        img_tensor = img_tensor.unsqueeze(0).to(device)
        label = torch.tensor([label]).to(device)
        opt.zero_grad()
        pred = model(img_tensor)
        loss = loss_fn(pred, label)
        loss.backward()
        opt.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_data)

    # 验证
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for img_tensor, label in val_data:
            img_tensor = img_tensor.unsqueeze(0).to(device)
            out = model(img_tensor)
            pred = torch.argmax(out, dim=1)
            total +=1
            if pred.item() == label:
                correct +=1
    acc = correct / total
    print(f"\nEpoch {epoch+1} | Loss:{avg_loss:.4f} | Val Acc:{acc:.4f}")

torch.save(model.state_dict(), "cat_dog_model.pth")
print("✅ 训练完成，模型保存为 cat_dog_model.pth")



