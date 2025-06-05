# 1. 图像预处理（聚焦锁孔区域）
import cv2
import numpy as np

def process_lock_area(image, rect):
    # 根据已知矩形框截取ROI区域
    x, y, w, h = rect
    roi = image[y:y+h, x:x+w]
    
    # 增强对比度预处理
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    
    return binary

# 2. 霍夫变换检测中心线
def detect_center_lines(binary):
    # 使用概率霍夫变换检测直线
    lines = cv2.HoughLinesP(binary, 1, np.pi/180, threshold=30, 
                            minLineLength=20, maxLineGap=5)
    
    # 筛选中心区域的线条（可根据实际调整坐标范围）
    center_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # 仅保留中间1/3高度的区域
            if abs((y1 + y2)/2 - binary.shape[0]/2) < binary.shape[0]/3:
                center_lines.append(line)
    
    return np.array(center_lines)

# 3. 计算偏转角度
def calculate_deviation(lines, rect):
    if len(lines) < 2:
        return None
        
    # 取两条最长线段的平均角度
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        angles.append(angle)
    
    # 计算角度均值作为偏转角度
    deviation_angle = np.mean(angles)
    
    # 转换为相对于矩形框横轴的角度（坐标系转换）
    return abs(deviation_angle)  # 返回绝对值表示偏转幅度

# 使用示例：

# 1. 读取图像
image = cv2.imread('/home/sophgo/Code/yichen/keyhole_detection/src/dataset/images/train/1748243267394.jpg')
# "1748243267394.jpg": {"x": 307, "y": 179, "w": 88, 
# "h": 88, "distance": 29.40, "is_lock_original": "False", "lock_angle": 65, "is_key_in": "False", "key_angle": 0},

# 2. 已知锁孔矩形框坐标 [x, y, width, height]
# rect_coords = [307,179,88,88]  # 示例坐标，需根据实际图像调整
rect = (307,179,88,88)  # 您已知的锁孔矩形框坐标
binary_image = process_lock_area(image, rect)
lines = detect_center_lines(binary_image)
deviation = calculate_deviation(lines, rect)

print(f"锁孔偏转角度: {deviation:.2f}")