import os, shutil, random

# 原始图片文件夹（解压后的全部猫狗图）
source_dir = r"raw_images/train"

# 目标目录
train_cats = r"dataset/train/cats"
train_dogs = r"dataset/train/dogs"
val_cats = r"dataset/val/cats"
val_dogs = r"dataset/val/dogs"

# 创建文件夹
for p in [train_cats, train_dogs, val_cats, val_dogs]:
    os.makedirs(p, exist_ok=True)

val_ratio = 0.2

# 分开筛选猫、狗图片
cat_imgs = [f for f in os.listdir(source_dir) if f.startswith("cat") and f.endswith((".jpg", ".png"))]
dog_imgs = [f for f in os.listdir(source_dir) if f.startswith("dog") and f.endswith((".jpg", ".png"))]

# 各取前50张
cat_imgs = cat_imgs[:50]
dog_imgs = dog_imgs[:50]

def copy_files(img_list, is_cat):
    random.seed(42)
    random.shuffle(img_list)
    val_num = int(len(img_list) * val_ratio)
    val_list = img_list[:val_num]
    train_list = img_list[val_num:]

    for fname in val_list:
        src = os.path.join(source_dir, fname)
        if is_cat:
            dst = os.path.join(val_cats, fname)
        else:
            dst = os.path.join(val_dogs, fname)
        shutil.copy(src, dst)

    for fname in train_list:
        src = os.path.join(source_dir, fname)
        if is_cat:
            dst = os.path.join(train_cats, fname)
        else:
            dst = os.path.join(train_dogs, fname)
        shutil.copy(src, dst)

copy_files(cat_imgs, is_cat=True)
copy_files(dog_imgs, is_cat=False)

print(f"猫图片总数：{len(cat_imgs)}，狗图片总数：{len(dog_imgs)}")
print("划分完成！")
