# 锁孔检测项目

## 概述
本项目旨在通过相机内参和标定数据，实现锁孔距离的估计。利用多项式拟合模型，结合图像中锁孔的像素尺寸，计算出锁孔与相机之间的实际距离。

## 项目结构
```plaintext
keyhole_detection/
├── .gitattributes
├── README.md
├── doc/
├── requirements.txt
├── result/
│   └── label.json
├── run.sh
├── slide/
│   └── box_statistics.png
└── src/
    ├── calibration/
    │   └── coefficients.json
    ├── onnx_model/
    ├── pth_model/
    │   ├── obb-yolo11x_epoch500_batch1_size640_model-x/
    │   └── yolo11x_epoch500_batch1_size640_model-x2/
    └── scripts/
        ├── __pycache__/
        ├── get_angle.py
        ├── get_distance.py
        ├── inference.py
        ├── process.py
        └── train.py
```

## 环境要求
- Python 3.11.11
- 所需依赖可在 `requirements.txt` 文件中查看：
```plaintext
ulturalytics==8.3.149
opencv-python==4.11.0.86
numpy==2.2.4
onnx
```

## 复现代码流程
### 1. 克隆仓库
```bash
git clone https://github.com/jiangyichen19/keyhole_detection.git
cd keyhole_detection
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 运行距离估计脚本
```bash
cd src/scripts
# 需要在该文件的 112行修改需要评估的文件路径：imgs_folder = '/home/sophgo/Code/yichen/keyhole_detection/src/dataset/images/val'
python process.py
```

## 核心脚本说明
- `get_distance.py`：包含 `DistanceEstimator` 类，用于添加标定样本、训练模型和估计距离。具体功能如下：
  - `add_calibration_sample`：添加标定样本。
  - `train_model`：训练距离估计模型并拟合多项式系数。
  - `estimate_distance_with_intrinsics`：使用相机内参估计距离。
  - `estimate_distance`：根据像素比例估计距离。
  - `load_coefficients`：加载已有的训练参数。

## 注意事项
- 在运行 `process.py` 前，请确保在该文件的 112 行修改需要评估的文件路径为 `imgs_folder = '/home/sophgo/Code/yichen/keyhole_detection/src/dataset/images/val'`。
- 若 `coefficients.json` 文件不存在，`load_coefficients` 方法会输出提示信息。
```

你可以将上述内容替换原有的 `README.md` 文件内容，以提供更详细和清晰的项目说明。 

        