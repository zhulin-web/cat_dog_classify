import os

source_dir = r"raw_images/train"

# 列出所有图片
all_files = os.listdir(source_dir)

cat_imgs = [f for f in all_files if f.startswith("cat") and f.endswith((".jpg", ".png"))]
dog_imgs = [f for f in all_files if f.startswith("dog") and f.endswith((".jpg", ".png"))]

# 只保留前50张
keep_cats = set(cat_imgs[:50])
keep_dogs = set(dog_imgs[:50])
keep_set = keep_cats | keep_dogs

# 遍历，不在保留集合里的就删掉
for fname in all_files:
    f_path = os.path.join(source_dir, fname)
    if os.path.isfile(f_path) and fname not in keep_set:
        os.remove(f_path)

print(f"保留猫图：{len(keep_cats)} 张，保留狗图：{len(keep_dogs)} 张")
print("筛选完成，源文件夹现在一共100张图片")
