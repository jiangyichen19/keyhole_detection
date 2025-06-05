import cv2
import numpy as np

# 1. 读取图像
image = cv2.imread('/home/sophgo/Code/yichen/keyhole_detection/src/dataset/images/train/1748243267394.jpg')
# "1748243267394.jpg": {"x": 307, "y": 179, "w": 88, 
# "h": 88, "distance": 29.40, "is_lock_original": "False", "lock_angle": 65, "is_key_in": "False", "key_angle": 0},

# 2. 已知锁孔矩形框坐标 [x, y, width, height]
rect_coords = [307,179,88,88]  # 示例坐标，需根据实际图像调整

# 3. 提取锁孔区域
x, y, w, h = rect_coords
lock_hole = image[y:y+h, x:x+w]

# 4. 灰度化 + 边缘检测
gray = cv2.cvtColor(lock_hole, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 30, 100, apertureSize=3)  # 降低高低阈值
cv2.imwrite('edges.jpg', edges) 



# 5. 霍夫直线检测（调整参数）
lines = cv2.HoughLinesP(edges, 
                       rho=1, 
                       theta=np.pi/180, 
                       threshold=30,  # 降低阈值
                       minLineLength=20,  # 缩短最小长度
                       maxLineGap=20)    # 增大间隔

# 新增调试信息输出
print(f"边缘检测像素统计：白色像素占比 {np.mean(edges>0)*100:.1f}%")
cv2.imwrite('lock_hole_roi.jpg', lock_hole)  # 保存原始ROI图像

if lines is not None:
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # 绘制检测到的直线
        cv2.line(lock_hole, (x1, y1), (x2, y2), (0,255,0), 2)
        
        # 计算线段角度（相对水平方向）
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        angles.append(angle if angle >=0 else angle + 180)  # 转换为0-180度
        
    # 计算平均角度
    avg_angle = np.mean(angles)
    
    # 输出结果
    print(f"📐 检测到 {len(lines)} 条直线")
    print(f"📏 平均偏转角度: {avg_angle:.2f}°")
    
    # 绘制角度标注
    cv2.putText(lock_hole, f"Angle: {avg_angle:.1f}°", 
               (10,30), cv2.FONT_HERSHEY_SIMPLEX, 
               0.7, (0,0,255), 2)

    # 保存结果
    cv2.imwrite('Lock_lines.jpg', lock_hole)
else:
    print("❌ 未检测到有效直线，请调整检测参数")

# # 5. Hough 圆检测
# circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 20,
#                            param1=50, param2=30,  # 可调整参数
#                            minRadius=10, maxRadius=50)  # 根据锁孔尺寸调整

# if circles is not None:
#     circles = np.uint16(np.around(circles))
#     for i in circles[0, :]:
#         # 6. 绘制检测到的圆
#         cv2.circle(lock_hole, (i[0], i[1]), i[2], (0, 255, 0), 2)
#         cv2.circle(lock_hole, (i[0], i[1]), 2, (0, 0, 255), 3)

#         # 7. 计算矩形中心与圆心偏差
#         rect_center_x = x + w // 2
#         rect_center_y = y + h // 2
#         deviation_x = i[0] - (w // 2)  # 相对于锁孔区域的偏移
#         deviation_y = i[1] - (h // 2)

#         # 8. 计算偏转角度
#         angle = np.degrees(np.arctan2(deviation_y, deviation_x))

#         # 9. 输出结果
#         print(f"✅ 锁孔归位状态: ", end="")
#         if abs(deviation_x) < 5 and abs(deviation_y) < 5:  # 偏差阈值可调
#             print("已归位")
#         else:
#             print(f"未归位 - 偏移量: 横向 {deviation_x} 像素, 纵向 {deviation_y} 像素")
#         print(f"🔍 偏转角度: {angle:.2f}°")

#     # 10. 显示结果
#     cv2.imwrite('Lock.jpg', lock_hole)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
# else:
#     print("❌ 未检测到锁孔圆心，请检查图像质量或调整参数")