import os
import cv2
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import torch
from PIL import Image
from train_model import AncientTextClassifier
import matplotlib
matplotlib.use("Agg")

class Predictor:
    def __init__(self, model_path="ancient_text_classifier.pth"):
        """初始化预测器"""
        self.classifier = AncientTextClassifier()
        self.model_path = model_path
        self.load_model()
    
    def load_model(self):
        """加载训练好的模型"""
        if os.path.exists(self.model_path):
            self.classifier.load_model(self.model_path)
            print(f"模型已成功加载: {self.model_path}")
            print(f"支持的类别: {self.classifier.class_names}")
            return True
        else:
            print(f"错误: 找不到模型文件 {self.model_path}")
            print("请先运行 train_model.py 训练模型")
            return False
    
    def preprocess_input_image(self, input_path):
        """预处理输入图像（与训练时的预处理保持一致）"""
        # 读取原图
        image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            print(f"警告：无法读取图像文件：{input_path}")
            return None
        
        # 与 data_prep.py 相同的预处理逻辑
        size = (128, 128)
        binary_thresh = 127
        
        # 通过计算像素数量判断是黑底白字还是白底黑字，并统一为白字黑底
        _, binary = cv2.threshold(image, binary_thresh, 255, cv2.THRESH_BINARY)
        if cv2.countNonZero(binary) > binary.size / 2:
            binary = cv2.bitwise_not(binary)  # 反转颜色，确保是白字黑底
        
        # 等比缩放+填充成固定大小（避免压缩变形）
        h, w = binary.shape
        scale = min(size[0]/w, size[1]/h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(binary, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # 创建背景图并居中粘贴字符图
        canvas = np.zeros(size, dtype=np.uint8)
        x_offset = (size[0] - new_w) // 2
        y_offset = (size[1] - new_h) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return canvas
    
    def predict_single_image(self, input_path):
        """预测单张图像"""
        if self.classifier.model is None:
            print("错误：模型未加载")
            return None, None
            
        # 预处理图像
        processed_img = self.preprocess_input_image(input_path)
        if processed_img is None:
            return None, None
        
        # 转换为PIL图像并使用模型进行预测
        img = Image.fromarray(processed_img)
        transform = self.classifier.get_transforms(is_training=False)
        img_tensor = transform(img).unsqueeze(0).to(self.classifier.device)
        
        # 预测
        self.classifier.model.eval()
        with torch.no_grad():
            output = self.classifier.model(img_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            probabilities = probabilities.cpu().numpy()[0] * 100  # 转换为百分比
        
        # 创建结果字典
        result = {}
        for i, class_name in enumerate(self.classifier.class_names):
            result[class_name] = float(probabilities[i])
        
        return result, processed_img
    
    def create_result_visualization(self, image_name, result, processed_img, output_dir):
        """创建预测结果的可视化图表"""
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 显示预处理后的图像
        ax1.imshow(processed_img, cmap='gray')
        ax1.set_title(f'预处理图像: {image_name}')
        ax1.axis('off')
        
        # 显示概率条形图
        countries = list(result.keys())
        probabilities = list(result.values())
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
        
        # 确保颜色数量与国家数量匹配
        if len(colors) < len(countries):
            colors = colors * ((len(countries) // len(colors)) + 1)
        colors = colors[:len(countries)]
        
        bars = ax2.bar(countries, probabilities, color=colors)
        ax2.set_title('各国文字识别概率')
        ax2.set_ylabel('概率 (%)')
        ax2.set_ylim(0, 100)
        
        # 在条形图上添加数值标签
        for bar, prob in zip(bars, probabilities):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{prob:.1f}%', ha='center', va='bottom')
        
        # 旋转x轴标签
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # 保存图表
        output_path = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}_result.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def predict_batch(self, input_dir="prediction/input", output_dir="prediction/output"):
        """批量预测文件夹中的所有图像"""
        if not os.path.exists(input_dir):
            print(f"错误: 输入文件夹不存在 {input_dir}")
            return []
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"创建输出文件夹: {output_dir}")
        
        # 支持的图像格式
        supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        
        # 收集所有图像文件
        image_files = []
        for filename in os.listdir(input_dir):
            if filename.lower().endswith(supported_extensions):
                image_files.append(filename)
        
        if not image_files:
            print(f"在 {input_dir} 中没有找到任何图像文件")
            return []
        
        print(f"找到 {len(image_files)} 个图像文件，开始批量预测...")
        
        # 存储所有结果
        all_results = []
        
        for i, filename in enumerate(image_files, 1):
            print(f"处理 {i}/{len(image_files)}: {filename}")
            
            input_path = os.path.join(input_dir, filename)
            result, processed_img = self.predict_single_image(input_path)
            
            if result is not None:
                # 找出最高概率的国家
                max_country = max(result, key=result.get)
                max_prob = result[max_country]
                
                # 创建可视化图表
                chart_path = self.create_result_visualization(
                    filename, result, processed_img, output_dir
                )
                
                # 保存预处理后的图像
                processed_img_path = os.path.join(
                    output_dir, f"{os.path.splitext(filename)[0]}_processed.png"
                )
                cv2.imwrite(processed_img_path, processed_img)
                
                # 记录结果
                result_record = {
                    'filename': filename,
                    'predicted_country': max_country,
                    'confidence': f"{max_prob:.1f}%",
                    'chart_path': os.path.basename(chart_path),
                    'processed_image_path': os.path.basename(processed_img_path)
                }
                
                # 添加各国概率
                for country, prob in result.items():
                    result_record[f"{country}_probability"] = f"{prob:.1f}%"
                
                all_results.append(result_record)
                
                print(f"  -> 预测结果: {max_country} ({max_prob:.1f}%)")
            else:
                print(f"  -> 跳过: 无法处理图像")
        
        # 保存汇总结果到CSV文件
        if all_results:
            df = pd.DataFrame(all_results)
            csv_path = os.path.join(output_dir, f"prediction_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            
            print(f"\n=== 批量预测完成 ===")
            print(f"处理图像数量: {len(all_results)}")
            print(f"结果保存到: {output_dir}")
            print(f"详细结果CSV: {csv_path}")
            
            # 打印汇总统计
            print(f"\n=== 预测统计 ===")
            predicted_countries = [r['predicted_country'] for r in all_results]
            for country in set(predicted_countries):
                count = predicted_countries.count(country)
                print(f"{country}: {count} 张图片")
        
        return all_results

def main():
    """主函数 - 提供交互式界面"""
    print("=== 古文字识别预测系统 (PyTorch版本) ===\n")
    
    # 检查模型文件
    model_files = ["ancient_text_classifier.pth", "best_model.pth"]
    model_file = None
    
    for f in model_files:
        if os.path.exists(f):
            model_file = f
            break
    
    if not model_file:
        print("错误: 找不到训练好的模型文件")
        print("请先运行 'python3 train_model.py' 训练模型")
        print("预期的模型文件: ancient_text_classifier.pth 或 best_model.pth")
        return
    
    # 初始化预测器
    predictor = Predictor(model_file)
    
    if not predictor.classifier.model:
        print("模型加载失败，退出程序")
        return
    
    while True:
        print("\n请选择操作:")
        print("1. 预测单张图片")
        print("2. 批量预测文件夹中的图片")
        print("3. 查看预测文件夹状态")
        print("4. 退出")
        
        choice = input("\n请输入选项 (1-4): ").strip()
        
        if choice == '1':
            # 单张图片预测
            image_path = input("请输入图片路径: ").strip()
            if os.path.exists(image_path):
                result, processed_img = predictor.predict_single_image(image_path)
                if result:
                    print(f"\n预测结果:")
                    for country, prob in sorted(result.items(), key=lambda x: x[1], reverse=True):
                        print(f"  {country}: {prob:.1f}%")
                    
                    # 可选：保存结果到输出文件夹
                    save = input("\n是否保存预测结果到输出文件夹? (y/n): ").lower()
                    if save == 'y':
                        if not os.path.exists("prediction/output"):
                            os.makedirs("prediction/output")
                        filename = os.path.basename(image_path)
                        chart_path = predictor.create_result_visualization(
                            filename, result, processed_img, "prediction/output"
                        )
                        print(f"结果图表已保存: {chart_path}")
            else:
                print("错误: 图片文件不存在")
        
        elif choice == '2':
            # 批量预测
            print("开始批量预测...")
            predictor.predict_batch()
        
        elif choice == '3':
            # 查看文件夹状态
            input_dir = "prediction/input"
            output_dir = "prediction/output"
            
            if os.path.exists(input_dir):
                input_files = [f for f in os.listdir(input_dir) 
                              if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
                print(f"输入文件夹 ({input_dir}): {len(input_files)} 个图片文件")
                if input_files:
                    print("  文件列表:", input_files[:5], "..." if len(input_files) > 5 else "")
            else:
                print(f"输入文件夹不存在: {input_dir}")
            
            if os.path.exists(output_dir):
                output_files = os.listdir(output_dir)
                print(f"输出文件夹 ({output_dir}): {len(output_files)} 个文件")
            else:
                print(f"输出文件夹不存在: {output_dir}")
        
        elif choice == '4':
            print("退出程序")
            break 

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已退出")
