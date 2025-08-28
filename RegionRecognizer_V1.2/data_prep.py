import cv2
import os
import numpy as np

def preprocess_image(input_path, output_path, size=(128, 128), binary_thresh=127):
    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f'警告：无法读取或跳过非图像文件：{input_path}')
        return
    _, binary = cv2.threshold(image, binary_thresh, 255, cv2.THRESH_BINARY)
    if cv2.countNonZero(binary) > binary.size / 2:
        binary = cv2.bitwise_not(binary)
    h, w = binary.shape
    scale = min(size[0] / w, size[1] / h)
    new_w, new_h = (int(w * scale), int(h * scale))
    resized = cv2.resize(binary, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros(size, dtype=np.uint8)
    x_offset = (size[0] - new_w) // 2
    y_offset = (size[1] - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    cv2.imwrite(output_path, canvas)
    print(f'处理完毕：{output_path}')

def process_state_images(state_input_dir, state_output_dir, size=(128, 128), binary_thresh=127):
    if not os.path.exists(state_output_dir):
        os.makedirs(state_output_dir)
        print(f'创建输出文件夹：{state_output_dir}')
    supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    if not os.path.isdir(state_input_dir):
        print(f"警告：在 {os.path.dirname(state_input_dir)} 中找不到 'input_images' 文件夹，已跳过。")
        return
    for filename in os.listdir(state_input_dir):
        if filename.lower().endswith(supported_extensions):
            input_path = os.path.join(state_input_dir, filename)
            output_path = os.path.join(state_output_dir, filename)
            preprocess_image(input_path, output_path, size, binary_thresh)
if __name__ == '__main__':
    base_data_dir = 'data'
    if not os.path.isdir(base_data_dir):
        print(f"错误：找不到基础数据目录 '{base_data_dir}'。请确保目录结构正确。")
    else:
        for state_name in os.listdir(base_data_dir):
            state_path = os.path.join(base_data_dir, state_name)
            if os.path.isdir(state_path):
                input_folder = os.path.join(state_path, 'input_images')
                output_folder = os.path.join(state_path, 'output_images')
                print(f'--- 开始处理国家：{state_name} ---')
                process_state_images(input_folder, output_folder)