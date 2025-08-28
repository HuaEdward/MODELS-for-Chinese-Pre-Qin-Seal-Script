import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import json
from PIL import Image

class AncientTextDataset(Dataset):
    """古文字数据集类"""
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # 转换为PIL图像
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class AncientTextCNN(nn.Module):
    """古文字识别CNN模型"""
    def __init__(self, num_classes=5):
        super(AncientTextCNN, self).__init__()
        
        # 特征提取层
        self.features = nn.Sequential(
            # 第一个卷积块
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # 第二个卷积块
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # 第三个卷积块
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # 第四个卷积块
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class AncientTextClassifier:
    def __init__(self, img_size=(128, 128), num_classes=5):
        self.img_size = img_size
        self.num_classes = num_classes
        self.model = None
        self.label_encoder = LabelEncoder()
        self.class_names = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
    def load_data(self, data_dir="data"):
        """加载各国文字数据"""
        images = []
        labels = []
        
        print("正在加载训练数据...")
        
        for state_name in os.listdir(data_dir):
            state_path = os.path.join(data_dir, state_name)
            if os.path.isdir(state_path):
                output_images_path = os.path.join(state_path, "output_images")
                
                if os.path.exists(output_images_path):
                    print(f"加载 {state_name} 的图片...")
                    
                    for filename in os.listdir(output_images_path):
                        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                            img_path = os.path.join(output_images_path, filename)
                            
                            # 读取图像
                            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                            if img is not None:
                                # 确保图像尺寸正确
                                img = cv2.resize(img, self.img_size)
                                
                                images.append(img)
                                labels.append(state_name)
        
        if len(images) == 0:
            raise ValueError("没有找到任何训练图片！请检查数据目录结构。")
        
        # 转换为numpy数组
        images = np.array(images)
        labels = np.array(labels)
        
        # 编码标签
        encoded_labels = self.label_encoder.fit_transform(labels)
        self.class_names = list(self.label_encoder.classes_)
        
        print(f"总共加载了 {len(images)} 张图片")
        print(f"类别: {self.class_names}")
        print(f"各类别数量: {dict(zip(*np.unique(labels, return_counts=True)))}")
        
        return images, encoded_labels
    
    def get_transforms(self, is_training=True):
        """获取数据变换"""
        if is_training:
            return transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.RandomRotation(10),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        else:
            return transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
    
    def train(self, images, labels, epochs=100, batch_size=16, validation_split=0.2, learning_rate=0.001):
        """训练模型"""
        # 分割训练和验证数据
        X_train, X_val, y_train, y_val = train_test_split(
            images, labels, test_size=validation_split, random_state=42, stratify=labels
        )
        
        # 创建数据集和数据加载器
        train_transform = self.get_transforms(is_training=True)
        val_transform = self.get_transforms(is_training=False)
        
        train_dataset = AncientTextDataset(X_train, y_train, train_transform)
        val_dataset = AncientTextDataset(X_val, y_val, val_transform)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 创建模型
        self.model = AncientTextCNN(num_classes=self.num_classes).to(self.device)
        
        # 损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        
        print("模型架构:")
        print(self.model)
        
        # 训练历史记录
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        
        best_val_acc = 0.0
        patience = 15
        patience_counter = 0
        
        print("开始训练模型...")
        
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                train_total += target.size(0)
                train_correct += (predicted == target).sum().item()
            
            # 验证阶段
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    loss = criterion(output, target)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(output.data, 1)
                    val_total += target.size(0)
                    val_correct += (predicted == target).sum().item()
            
            # 计算平均损失和准确率
            train_loss_avg = train_loss / len(train_loader)
            train_acc = 100. * train_correct / train_total
            val_loss_avg = val_loss / len(val_loader)
            val_acc = 100. * val_correct / val_total
            
            # 记录历史
            train_losses.append(train_loss_avg)
            train_accuracies.append(train_acc)
            val_losses.append(val_loss_avg)
            val_accuracies.append(val_acc)
            
            # 更新学习率
            scheduler.step()
            
            # 打印进度
            if epoch % 5 == 0 or epoch == epochs - 1:
                print(f'Epoch {epoch+1}/{epochs}:')
                print(f'  训练 - Loss: {train_loss_avg:.4f}, Acc: {train_acc:.2f}%')
                print(f'  验证 - Loss: {val_loss_avg:.4f}, Acc: {val_acc:.2f}%')
                print(f'  学习率: {optimizer.param_groups[0]["lr"]:.6f}')
            
            # 早停检查
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # 保存最佳模型
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f'早停：验证准确率在 {patience} 个epochs内没有改善')
                break
        
        # 加载最佳模型
        self.model.load_state_dict(torch.load('best_model.pth'))
        
        # 返回训练历史
        history = {
            'train_loss': train_losses,
            'train_accuracy': train_accuracies,
            'val_loss': val_losses,
            'val_accuracy': val_accuracies
        }
        
        print(f'训练完成！最佳验证准确率: {best_val_acc:.2f}%')
        return history
    
    def plot_training_history(self, history):
        """绘制训练历史"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # 准确率曲线
        ax1.plot(history['train_accuracy'], label='Training Accuracy')
        ax1.plot(history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy (%)')
        ax1.legend()
        ax1.grid(True)
        
        # 损失曲线
        ax2.plot(history['train_loss'], label='Training Loss')
        ax2.plot(history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def save_model(self, model_path="ancient_text_classifier.pth"):
        """保存模型"""
        if self.model is not None:
            # 保存模型权重
            torch.save(self.model.state_dict(), model_path)
            
            # 保存模型信息
            model_info = {
                'class_names': self.class_names,
                'img_size': self.img_size,
                'num_classes': self.num_classes,
                'model_architecture': str(self.model)
            }
            
            with open('model_info.json', 'w', encoding='utf-8') as f:
                json.dump(model_info, f, ensure_ascii=False, indent=2)
            
            print(f"模型已保存到: {model_path}")
            print("模型信息已保存到: model_info.json")
        else:
            print("错误：没有训练好的模型可以保存")
    
    def load_model(self, model_path="ancient_text_classifier.pth"):
        """加载模型"""
        if os.path.exists(model_path):
            # 加载模型信息
            if os.path.exists('model_info.json'):
                with open('model_info.json', 'r', encoding='utf-8') as f:
                    model_info = json.load(f)
                
                self.class_names = model_info['class_names']
                self.img_size = tuple(model_info['img_size'])
                self.num_classes = model_info['num_classes']
                
                # 重建标签编码器
                self.label_encoder.classes_ = np.array(self.class_names)
            
            # 创建模型并加载权重
            self.model = AncientTextCNN(num_classes=self.num_classes).to(self.device)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            
            print(f"模型已从 {model_path} 加载")
        else:
            print(f"错误：找不到模型文件 {model_path}")
    
    def predict(self, image_path):
        """预测单张图片"""
        if self.model is None:
            print("错误：请先训练或加载模型")
            return None
        
        # 读取并预处理图像
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"错误：无法读取图像 {image_path}")
            return None
        
        # 预处理
        img = cv2.resize(img, self.img_size)
        img = Image.fromarray(img)
        
        transform = self.get_transforms(is_training=False)
        img_tensor = transform(img).unsqueeze(0).to(self.device)
        
        # 预测
        self.model.eval()
        with torch.no_grad():
            output = self.model(img_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            probabilities = probabilities.cpu().numpy()[0] * 100  # 转换为百分比
        
        # 创建结果字典
        result = {}
        for i, class_name in enumerate(self.class_names):
            result[class_name] = float(probabilities[i])
        
        return result

def main():
    """主函数"""
    print("=== 古文字识别模型训练系统 (PyTorch版本) ===")
    
    # 初始化分类器
    classifier = AncientTextClassifier(img_size=(128, 128), num_classes=5)
    
    try:
        # 加载数据
        images, labels = classifier.load_data("data")
        
        # 训练模型
        history = classifier.train(
            images, labels, 
            epochs=100, 
            batch_size=16, 
            validation_split=0.2,
            learning_rate=0.001
        )
        
        # 绘制训练历史
        classifier.plot_training_history(history)
        
        # 保存模型
        classifier.save_model()
        
        print("\n=== 训练完成 ===")
        print("可以使用以下代码进行预测:")
        print("""
# 加载模型并预测
from train_model import AncientTextClassifier
classifier = AncientTextClassifier()
classifier.load_model()
result = classifier.predict('path/to/your/image.png')
print(result)
        """)
        
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()