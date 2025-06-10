import os
import time
import json
from ultralytics import YOLO
from get_distance import DistanceEstimator
import numpy as np



#
#参数:
#   img_path: 要识别的图片的路径
#
#返回:
#   返回结果为各赛题中要求的识别结果，具体格式可参考提供压缩包中的 “图片对应输出结果.txt” 中一张图片对应的结果
#
def process_img(img_path):
    # 进行推理
        # 加载模型
    model = YOLO(r"../pth_model/yolo11x_epoch500_batch1_size640_model-x2/weights/best.pt")  # pretrained YOLO11n model
    obb_model = YOLO(r"../pth_model/obb-yolo11x_epoch500_batch1_size640_model-x/weights/best.pt")  # 加载 obb 模型
    estimator = DistanceEstimator()

    coefficients_path = r"../calibration/coefficients.json"
    # 使用新添加的函数加载系数
    estimator.load_coefficients(coefficients_path)
    
    
    results = model([img_path])
    result = results[0]
    boxes = result.boxes  # Boxes object for bounding box outputs
    is_key_in = False
    x, y, w, h = None, None, None, None
    distance = 0
    lock_angle = 0
    key_angle = 0
    is_lock_original = True

    if len(boxes) > 0:
        class_id = int(boxes.cls[0].item())
        is_key_in = class_id != 0

        # 获取第一张图片的尺寸
        img = result.orig_img
        img_height, img_width = img.shape[:2]

        # 获取边界框信息，假设使用第一个检测框
        box = boxes[0].cpu().numpy()
        xyxy = box.xyxy[0]  # x1, y1, x2, y2 格式
        x = int(xyxy[0])
        y = int(xyxy[1])
        w = int(xyxy[2] - xyxy[0])
        h = int(xyxy[3] - xyxy[1])

        # 计算距离
        real_size = 28
        distance = estimator.estimate_distance_with_intrinsics(
            box_size=(w, h),
            real_size=real_size
        )
        distance = float(distance)

        if distance > 35:
            # 裁剪出检测框对应的图像区域
            x1, y1, x2, y2 = map(int, xyxy)
            cropped_img = img[y1:y2, x1:x2]

            # 调整图像大小为 640x640
            import cv2
            resized_img = cv2.resize(cropped_img, (640, 640))

            # 保存临时图像文件（如果模型需要文件路径输入）
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                cv2.imwrite(tmp.name, resized_img)
                # 使用调整大小后的图像执行 obb 角度检测
                obb_results = obb_model(tmp.name)
                os.unlink(tmp.name)

            for obb_result in obb_results:
                xywhr = obb_result.obb.xywhr  # center-x, center-y, width, height, angle (radians)
                # 将弧度转换为角度
                lock_angle = xywhr[..., -1].cpu().numpy() * (180 / 3.141592653589793) if xywhr.numel() > 0 else 0
                key_angle = lock_angle  # 假设锁和钥匙角度相同，可根据实际情况修改
                if lock_angle >= 10:
                    is_lock_original = False
                
                # is_lock_original = lock_angle == 0 and key_angle == 0

    # 构建标注数据
    label_data = {
        os.path.basename(img_path): {
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "distance": distance,
            "is_lock_original": is_lock_original,
            "lock_angle": lock_angle,
            "is_key_in": is_key_in,
            "key_angle": key_angle
        }
    }
    return label_data

#
#以下代码仅作为选手测试代码时使用，仅供参考，可以随意修改
#但是最终提交代码后，process.py文件是作为模块进行调用，而非作为主程序运行
#因此提交时请根据情况删除不必要的额外代码
#
if __name__ == '__main__':
    

     ### !!!!!!!!!!!!!!!!!!!!!!!!!!!替换为你要测试的图片路径 ###
    imgs_folder = r'/home/sophgo/Code/yichen/keyhole_detection/src/dataset/images/val'
    img_paths = [os.path.join(imgs_folder, f) for f in os.listdir(imgs_folder) if f.endswith(('.jpg', '.png'))]
    def now():
        return int(time.time()*1000)
    last_time = 0
    count_time = 0
    max_time = 0
    min_time = now()
    all_label_data = {}

    for img_path in img_paths:
        print(img_path,':')
        last_time = now()
        result = process_img(img_path)
        run_time = now() - last_time
        print('result:\n',result)
        print('run time: ', run_time, 'ms')
        print()
        count_time += run_time
        if run_time > max_time:
            max_time = run_time
        if run_time < min_time:
            min_time = run_time
        all_label_data.update(result)

    # 保存结果到 label.json
    for img_name, data in all_label_data.items():
        if isinstance(data['distance'], np.float64):
            data['distance'] = float(data['distance'])
        if isinstance(data['lock_angle'], np.ndarray):
            data['lock_angle'] = data['lock_angle'].tolist()
        if isinstance(data['key_angle'], np.ndarray):
            data['key_angle'] = data['key_angle'].tolist()
            
    with open('../../result/label.json', 'w') as f:
        json.dump(all_label_data, f)

    print('\n')
    print('avg time: ',int(count_time/len(img_paths)),'ms')
    print('max time: ',max_time,'ms')
    print('min time: ',min_time,'ms')