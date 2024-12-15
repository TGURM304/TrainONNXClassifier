import cv2
import numpy as np
import onnxruntime as ort
import os

# ONNX模型路径
model_path = 'best.onnx'  # 请替换为实际模型路径

# 定义类别标签
class_labels = ['1', '2', '3', '4', 'base',  'qsz', 'sb', 'null']

# 初始化ONNX模型
session = ort.InferenceSession(model_path)

# 打开摄像头
cap = cv2.VideoCapture(0)

# 检查摄像头是否打开
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

# 创建保存ROI图像的文件夹（如果不存在）
roi_folder = 'ROI'
if not os.path.exists(roi_folder):
    os.makedirs(roi_folder)

# 用来保存图像的计数器
image_counter = 0

while True:
    # 读取视频流中的一帧
    ret, frame = cap.read()

    if not ret:
        print("无法读取帧")
        break

    # 获取帧的尺寸
    height, width, _ = frame.shape

    # 计算中心区域的边长 (正方形ROI)
    roi_size = 128
    top_left_x = (width - roi_size) // 2
    top_left_y = (height - roi_size) // 2

    # 提取中心正方形区域 (ROI)
    roi = frame[top_left_y:top_left_y + roi_size, top_left_x:top_left_x + roi_size]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    ret, roi_binary = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    roi_bgr = cv2.cvtColor(roi_binary, cv2.COLOR_GRAY2BGR)

    # 预处理图像
    # 将图像调整为模型要求的输入尺寸
    input_image = cv2.resize(roi_bgr, (64, 64))  # 假设模型的输入大小为64x64
    input_image = input_image.astype(np.float32) / 255.0  # 归一化到0-1之间
    input_image = np.expand_dims(input_image, axis=0)  # 添加batch维度

    # 获取模型输入的名字
    input_name = session.get_inputs()[0].name

    # 进行推理
    result = session.run(None, {input_name: input_image})

    # 获取预测结果，假设是分类概率，获取最大概率的类别
    predicted_class = np.argmax(result[0])

    # 显示分类结果
    label = class_labels[predicted_class]

    # 在图像上绘制ROI区域和分类标签
    cv2.rectangle(frame, (top_left_x, top_left_y), (top_left_x + roi_size, top_left_y + roi_size), (0, 255, 0), 2)
    cv2.putText(frame, f"Predicted: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # 显示结果
    cv2.imshow("Classification", frame)
    cv2.imshow("ROI", roi_bgr)

    # 检测按键
    key = cv2.waitKey(1) & 0xFF

    # 如果按下空格键，保存当前的ROI图像到ROI文件夹中
    if key == ord(' '):
        roi_filename = os.path.join(roi_folder, f'roi_{image_counter}.png')
        cv2.imwrite(roi_filename, roi_bgr)  # 保存ROI图像
        print(f"保存ROI图像: {roi_filename}")
        image_counter += 1

    # 按 'q' 键退出
    if key == ord('q'):
        break

# 释放摄像头并关闭所有窗口
cap.release()
cv2.destroyAllWindows()
