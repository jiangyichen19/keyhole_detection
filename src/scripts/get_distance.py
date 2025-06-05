import numpy as np
from pathlib import Path
import json
import os

class DistanceEstimator:
    def __init__(self):
        self.calibration_data = []
        
    def add_calibration_sample(self, box_size, image_size, real_distance):
        """添加标定样本"""
        # box_size: (w, h), image_size: (W, H), real_distance: float
        box_ratio = box_size[0] * box_size[1] / (image_size[0] * image_size[1])
        self.calibration_data.append((box_ratio, real_distance))
        
    def train_model(self):
            """训练距离估计模型"""
            ratios = np.array([x[0] for x in self.calibration_data])
            distances = np.array([x[1] for x in self.calibration_data])
            
            # 新增考虑长宽比的非线性模型：distance ≈ k * (w/h)^a / ratio^b
            X = np.column_stack([np.log(ratios), np.ones_like(ratios)])
            theta = np.linalg.lstsq(X, np.log(distances), rcond=None)[0]
            self.a = theta[0]
            self.b = theta[1]
        
    def estimate_distance(self, box_size, image_size):
        """估计距离"""
        w, h = box_size
        ratio = w * h / (image_size[0] * image_size[1])
        return np.exp(self.b) * (w/h)**self.a / ratio**0.5

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
        json.dump({"k": estimator.k}, f)
        # json.dump({"k": estimator.k}, f)
    
    # 示例测试（使用最后一条数据验证）
    test_data = list(annotations.values())[-1]
    new_distance = estimator.estimate_distance(
        box_size=(test_data['w'], test_data['h']),
        image_size=(W, H)
    )
    
    print(f"标定完成！系数k={estimator.k:.2f}")
    # /home/sophgo/Code/yichen/detection/dataset/test/data1/visualization_results/vis_1748242785054.jpg
    print(f"验证数据：{test_data} => 估计距离: {new_distance:.2f}厘米")

if __name__ == "__main__":
    main()
    # W = 640
    # H = 480
    # estimator = DistanceEstimator()
    # # {"x": 307, "y": 179, "w": 88, "h": 88, "distance": 29.40, 
    # new_distance = estimator.estimate_distance(
    #     box_size=(88, 88),
    #     image_size=(W, H)
    # )
    
    # # print(f"标定完成！系数k={estimator.k:.2f}")
    # # /home/sophgo/Code/yichen/detection/dataset/test/data1/visualization_results/vis_1748242785054.jpg
    # print(f"验证数据：29.40,  => 估计距离: {new_distance:.2f}厘米")