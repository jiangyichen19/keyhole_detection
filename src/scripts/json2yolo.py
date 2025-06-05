import json
import os
import shutil
import random
from PIL import Image

def convert_to_yolo(json_path, img_dir, output_dir, train_ratio=0.8):
    # 创建标准YOLO目录结构
    dirs = ['images/train', 'images/val', 'labels/train', 'labels/val']
    for d in dirs:
        os.makedirs(os.path.join(output_dir, d), exist_ok=True)

    # 加载标注文件
    with open(json_path) as f:
        annotations = json.load(f)

    # 获取所有图片并打乱顺序
    all_images = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg'))]
    random.shuffle(all_images)
    
    # 划分训练/验证集
    split_idx = int(len(all_images) * train_ratio)
    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]

    # 处理训练集
    for img_name in train_images:
        process_image(img_name, img_dir, output_dir, annotations, 'train')

    # 处理验证集
    for img_name in val_images:
        process_image(img_name, img_dir, output_dir, annotations, 'val')

def process_image(img_name, img_dir, output_dir, annotations, dataset_type):
    try:
        img_path = os.path.join(img_dir, img_name)
        
        # 获取图片尺寸
        with Image.open(img_path) as img:
            img_w, img_h = img.size

        # 复制图片到对应目录
        shutil.copy(img_path, 
                   os.path.join(output_dir, 'images', dataset_type, img_name))
        
        # 生成标签文件
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(output_dir, 'labels', dataset_type, label_name)
        
        # 写入YOLO格式标注
        with open(label_path, 'w') as f:
            if img_name in annotations:
                ann = annotations[img_name]
                # 转换坐标
                x_center = (ann['x'] + ann['w']/2) / img_w
                y_center = (ann['y'] + ann['h']/2) / img_h
                width = ann['w'] / img_w
                height = ann['h'] / img_h
                
                # 判断类别
                class_id = 1 if ann.get('is_key_in', 'False') == 'True' else 0
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    except Exception as e:
        print(f"处理图片 {img_name} 时出错: {str(e)}")

if __name__ == "__main__":
    json_path = "/home/sophgo/Code/yichen/detection/dataset/test/data1/label.json"
    img_dir = "/home/sophgo/Code/yichen/detection/dataset/test/data1"
    output_dir = "/home/sophgo/Code/yichen/keyhole_detection/src/dataset"  # 调整为README中的路径
    
    convert_to_yolo(json_path, img_dir, output_dir)
    print("数据集划分完成！目录结构已创建为：")
    print("""
    dataset/
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/
    """)