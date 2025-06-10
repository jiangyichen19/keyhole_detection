import sys
# 添加 process.py 所在目录到系统路径
sys.path.append('/home/sophgo/Code/yichen/calibration/keyhole_detection/src/scripts')

from process import process_img,init_model
import os

# 替换为你要测试的图片路径
img_path = '/home/sophgo/Code/yichen/detection/dataset/test/data1/visualization_results/vis_1748242655899.jpg'

if __name__ == '__main__':
    model, obb_model,estimator = init_model()
    if os.path.exists(img_path):
        result = process_img(model, obb_model,estimator ,img_path)
        print('识别结果:\n', result)
    else:
        print(f'指定的图片路径 {img_path} 不存在，请检查路径。')