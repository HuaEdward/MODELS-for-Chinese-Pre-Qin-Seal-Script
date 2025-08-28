import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from single_character_predict import Predictor

class InscriptionPredictor:

    def __init__(self, model_path='best_model.pth'):
        self.predictor = Predictor(model_path)

    def predict_inscription(self, inscription_images_dir):
        if not os.path.isdir(inscription_images_dir):
            print(f'错误：找不到目录 {inscription_images_dir}')
            return (None, None)
        supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        image_files = [f for f in os.listdir(inscription_images_dir) if f.lower().endswith(supported_extensions)]
        if not image_files:
            print(f'在 {inscription_images_dir} 中没有找到任何图像文件')
            return (None, None)
        all_predictions = []
        detailed_results = []
        for filename in image_files:
            image_path = os.path.join(inscription_images_dir, filename)
            result, _ = self.predictor.predict_single_image(image_path)
            if result:
                all_predictions.append(result)
                prediction_details = {'image': filename, **result}
                detailed_results.append(prediction_details)
        if not all_predictions:
            print('未能对任何图像进行预测。')
            return (None, None)
        df = pd.DataFrame(all_predictions)
        mean_probabilities = df.mean().to_dict()
        total_prob = sum(mean_probabilities.values())
        normalized_probabilities = {k: v / total_prob * 100 for k, v in mean_probabilities.items()} if total_prob > 0 else mean_probabilities
        return (dict(sorted(normalized_probabilities.items(), key=lambda item: item[1], reverse=True)), detailed_results)

    def save_results(self, output_dir, inscription_name, overall_probabilities, detailed_results):
        os.makedirs(output_dir, exist_ok=True)
        self._save_plot(output_dir, inscription_name, overall_probabilities)
        self._save_csv(output_dir, inscription_name, detailed_results)

    def _save_plot(self, output_dir, inscription_name, probabilities):
        plt.figure(figsize=(10, 6))
        countries = list(probabilities.keys())
        probs = list(probabilities.values())
        bars = plt.bar(countries, probs, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57'])
        plt.title(f'碑文整体预测结果: {inscription_name}')
        plt.ylabel('概率 (%)')
        plt.ylim(0, 100)
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2.0, height + 1, f'{height:.1f}%', ha='center', va='bottom')
        plot_path = os.path.join(output_dir, 'input_overall_prediction.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'总体预测图表已保存至: {plot_path}')

    def _save_csv(self, output_dir, inscription_name, detailed_results):
        if not detailed_results:
            return
        df = pd.DataFrame(detailed_results)
        for col in df.columns:
            if col != 'image':
                df[col] = df[col].apply(lambda x: f'{x:.2f}%')
        csv_path = os.path.join(output_dir, 'input_detailed_predictions.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f'详细预测结果已保存至: {csv_path}')

def main():
    print('=== 批量碑文预测系统 ===')
    base_input_dir = 'inscription_prediction/input'
    base_output_dir = 'inscription_prediction/output'
    if not os.path.isdir(base_input_dir) or not os.listdir(base_input_dir):
        print(f"错误：输入目录 '{base_input_dir}' 不存在或为空。")
        print('请在其中为每个碑文创建子目录。')
        os.makedirs(base_input_dir, exist_ok=True)
        return
    predictor = InscriptionPredictor()
    for inscription_name in os.listdir(base_input_dir):
        current_input_dir = os.path.join(base_input_dir, inscription_name)
        if os.path.isdir(current_input_dir):
            print(f'\n--- 正在处理碑文: {inscription_name} ---')
            overall_probabilities, detailed_results = predictor.predict_inscription(current_input_dir)
            if overall_probabilities:
                current_output_dir = os.path.join(base_output_dir, inscription_name)
                print(f"'{inscription_name}' 的整体预测结果:")
                for country, probability in overall_probabilities.items():
                    print(f'  - {country}: {probability:.2f}%')
                predictor.save_results(current_output_dir, inscription_name, overall_probabilities, detailed_results)
if __name__ == '__main__':
    main()