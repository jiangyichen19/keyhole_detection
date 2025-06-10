import numpy as np
from pathlib import Path
import json
import os

# 从 get_angle3.py 中获取相机内参
K = np.array([
    [546.216, 0, 289.049],    # f_x, 0, cx
    [0, 545.736, 259.409],    # 0, f_y, cy
    [0, 0, 1]                 # 0, 0, 1
])

class DistanceEstimator:
    def __init__(self):
        self.calibration_data = []
        self.poly_coeffs = None  # 多项式系数

    def add_calibration_sample(self, box_size, image_size, real_distance):
        """添加标定样本"""
        w_pixels = box_size[0]
        self.calibration_data.append((w_pixels, real_distance))

    def train_model(self):
        """训练距离估计模型并拟合多项式系数"""
        w_pixels_list = np.array([x[0] for x in self.calibration_data])
        real_distances = np.array([x[1] for x in self.calibration_data])

        f = (K[0, 0] + K[1, 1]) / 2
        W = 28  # 假设锁孔真实宽度

        # 使用多项式拟合
        degree = 3  # 多项式的阶数，可以根据实际情况调整
        self.poly_coeffs = np.polyfit(w_pixels_list, real_distances, degree)

    def estimate_distance_with_intrinsics(self, box_size, real_size):
        """使用相机内参估计距离"""
        w_pixels = box_size[0]
        f = (K[0, 0] + K[1, 1]) / 2
        W = real_size

        # 根据多项式模型计算距离
        distance = np.polyval(self.poly_coeffs, w_pixels)
        return distance

    def estimate_distance(self, box_size, image_size):
        """估计距离"""
        # 暂时保留原逻辑，可根据需要调整
        w, h = box_size
        ratio = w * h / (image_size[0] * image_size[1])
        return np.exp(getattr(self, 'b', 0)) * (w/h)**getattr(self, 'a', 0) / ratio**0.5

    def load_coefficients(self, coefficients_path):
        """加载已有的训练参数"""
        try:
            with open(coefficients_path) as f:
                data = json.load(f)
                if data.get("poly_coeffs"):
                    self.poly_coeffs = np.array(data["poly_coeffs"])
        except FileNotFoundError:
            print(f"未找到系数文件: {coefficients_path}")

def main():
    W = 640
    H = 480
    estimator = DistanceEstimator()

    # 从JSON文件加载标定数据
    json_path = "/home/sophgo/Code/yichen/detection/dataset/test/data1/label.json"
    with open(json_path) as f:
        annotations = json.load(f)

    # 添加所有标定样本
    for img_name, data in annotations.items():
        estimator.add_calibration_sample(
            box_size=(data['w'], data['h']),
            image_size=(W, H),
            real_distance=data['distance']
        )

    # 训练并保存模型系数
    estimator.train_model()
    os.makedirs("../calibration", exist_ok=True)  # 添加目录创建语句
    with open("../calibration/coefficients.json", "w") as f:
        json.dump({
            "poly_coeffs": estimator.poly_coeffs.tolist() if estimator.poly_coeffs is not None else None
        }, f)

    # 假设锁孔的真实宽度（单位：mm），需要根据实际情况修改
    real_size = 28

    errors = []
    # 使用所有数据进行测试
    for data in annotations.values():
        # 使用相机内参估计距离
        estimated_distance = estimator.estimate_distance_with_intrinsics(
            box_size=(data['w'], data['h']),
            real_size=real_size
        )
        # 计算误差
        error = abs(estimated_distance - data['distance'])
        errors.append(error)
        print(error, data['distance'])

    errors = np.array(errors)
    max_error = np.max(errors)
    min_error = np.min(errors)
    avg_error = np.mean(errors)

    print(f"最大误差: {max_error:.2f} mm")
    print(f"最小误差: {min_error:.2f} mm")
    print(f"平均误差: {avg_error:.2f} mm")

if __name__ == "__main__":
    main()