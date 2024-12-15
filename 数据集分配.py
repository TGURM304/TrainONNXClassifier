import os
import shutil
import random

# 定义train和val文件夹路径
train_dir = './train'
val_dir = './val'

# 确保val文件夹存在
if not os.path.exists(val_dir):
    os.makedirs(val_dir)


def balance_categories(train_path, val_path, val_ratio=0.2):
    # 获取该类别文件夹中的所有图片文件，忽略 Thumbs.db
    train_images = [img for img in os.listdir(train_path) if
                    os.path.isfile(os.path.join(train_path, img)) and img.lower() != 'thumbs.db']
    val_images = [img for img in os.listdir(val_path) if
                  os.path.isfile(os.path.join(val_path, img)) and img.lower() != 'thumbs.db']

    # 计算总图片数量
    total_images = len(train_images) + len(val_images)

    # 根据比例计算目标数量
    target_val_count = int(total_images * val_ratio)
    target_train_count = total_images - target_val_count

    # 如果train文件夹中的图片数量大于目标数量，从train移动到val
    if len(train_images) > target_train_count:
        images_to_move = random.sample(train_images, len(train_images) - target_train_count)
        for img in images_to_move:
            src_path = os.path.join(train_path, img)
            dst_path = os.path.join(val_path, img)
            try:
                shutil.move(src_path, dst_path)
            except Exception as e:
                print(f"无法移动图片 {img}: {e}")

    # 如果val文件夹中的图片数量大于目标数量，从val移动到train
    elif len(val_images) > target_val_count:
        images_to_move = random.sample(val_images, len(val_images) - target_val_count)
        for img in images_to_move:
            src_path = os.path.join(val_path, img)
            dst_path = os.path.join(train_path, img)
            try:
                shutil.move(src_path, dst_path)
            except Exception as e:
                print(f"无法移动图片 {img}: {e}")


def count_images(directory):
    total_count = 0
    category_counts = {}
    for category in os.listdir(directory):
        category_path = os.path.join(directory, category)
        if os.path.isdir(category_path):
            images = [img for img in os.listdir(category_path) if
                      os.path.isfile(os.path.join(category_path, img)) and img.lower() != 'thumbs.db']
            category_counts[category] = len(images)
            total_count += len(images)
    return total_count, category_counts


# 遍历train文件夹中的每个类别文件夹
for category in os.listdir(train_dir):
    category_train_path = os.path.join(train_dir, category)
    category_val_path = os.path.join(val_dir, category)

    if os.path.isdir(category_train_path):
        # 确保val文件夹中的类别文件夹也存在
        if not os.path.exists(category_val_path):
            os.makedirs(category_val_path)

        # 平衡每个类别文件夹中的图片数量，设定val占20%
        balance_categories(category_train_path, category_val_path, val_ratio=0.25)

# 统计并输出train和val文件夹中的图片数量
train_total, train_counts = count_images(train_dir)

print(f"Train文件夹总图片数量: {train_total}")
for category, count in train_counts.items():
    print(f"  {category}: {count}")

print("所有类别的图片已按比例分置")
