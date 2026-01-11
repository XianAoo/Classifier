import os
import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

# --- 全局配置类 (路径与超参) ---
class Config:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Classifier/
    DATASET_DIR = os.path.join(os.path.dirname(BASE_DIR), 'Dataset') # Project/Dataset/
    
    # 文件路径
    SCALER_PATH = os.path.join(DATASET_DIR, 'scaler.pkl')
    FEATURE_NAMES_PATH = os.path.join(DATASET_DIR, 'feature_names.pkl')
    CSV_BOTNET = os.path.join(DATASET_DIR, 'Botnet.csv')
    CSV_BENIGN = os.path.join(DATASET_DIR, 'BenignTraffic.csv')
    JSON_BOTNET = os.path.join(DATASET_DIR, 'Botnet.json')
    JSON_BENIGN = os.path.join(DATASET_DIR, 'BenignTraffic.json')

    # 默认训练参数
    BATCH_SIZE = 64
    EPOCHS = 15
    LR = 0.001
    SEQ_LEN = 32  # 序列长度
    MAX_PACKET_VALUE = 1514
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 工具类 1: 流量统计特征数据管理器 (CSV) ---
class FlowDataManager:
    def __init__(self):
        if not os.path.exists(Config.SCALER_PATH):
            raise FileNotFoundError(f"找不到 Scaler: {Config.SCALER_PATH}")
        self.scaler = joblib.load(Config.SCALER_PATH)
        self.feature_names = joblib.load(Config.FEATURE_NAMES_PATH)
        print(f"[FlowDataManager] 加载配置成功: {len(self.feature_names)} 个特征")

    def load_data(self):
        """读取 CSV -> 对齐列 -> 标准化 -> 返回 Numpy 数组"""
        # 1. 读取
        df_bot = pd.read_csv(Config.CSV_BOTNET)
        df_ben = pd.read_csv(Config.CSV_BENIGN)
        
        # 2. 标签
        df_ben['Label'] = 0
        df_bot['Label'] = 1
        
        # 3. 合并
        df = pd.concat([df_bot, df_ben], ignore_index=True)
        
        # 4. 列对齐 (关键)
        df.columns = df.columns.str.strip()
        X_raw = df[self.feature_names].values
        y = df['Label'].values
        
        # 5. 处理 NaN
        X_raw = np.nan_to_num(X_raw)

        # 6. 标准化 (Transform only)
        X_scaled = self.scaler.transform(X_raw)
        
        # 7. 切分
        return train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    def get_dataloaders(self, batch_size=Config.BATCH_SIZE):
        """获取 PyTorch DataLoader"""
        X_train, X_test, y_train, y_test = self.load_data()
        
        train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
        test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        
        return train_loader, test_loader, X_train.shape[1]

# --- 工具类 2: 序列特征数据管理器 (JSON) ---
class SequenceDataManager:
    def __init__(self, max_len=Config.SEQ_LEN):
        self.max_len = max_len
        self.vocab_size = 0

    def _extract(self, filepath, label):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        seqs = []
        for flow in data:
            # [修改前]: raw = [pkt['packet_size'] for pkt in flow['packets']]
        
            # [修改后]: 读取时直接截断数值
            raw = [min(pkt['packet_size'], Config.MAX_PACKET_VALUE) for pkt in flow['packets']]
            
            # 下面保持不变 (截断长度/填充)
            if len(raw) >= self.max_len:
                new_seq = raw[:self.max_len]
            else:
                new_seq = raw + [0] * (self.max_len - len(raw))
            seqs.append(new_seq)
            
        return np.array(seqs, dtype=np.int64), np.full(len(seqs), label, dtype=np.int64)

    def load_data(self):
        """读取 JSON -> 截断/填充 -> 返回 Numpy 数组"""
        X_ben, y_ben = self._extract(Config.JSON_BENIGN, 0)
        X_bot, y_bot = self._extract(Config.JSON_BOTNET, 1)
        
        X = np.concatenate([X_bot, X_ben], axis=0)
        y = np.concatenate([y_bot, y_ben], axis=0)
        
        # 更新 vocab_size
        self.vocab_size = int(np.max(X)) + 1 + 10
        
        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    def get_dataloaders(self, batch_size=Config.BATCH_SIZE):
        """获取 PyTorch DataLoader (LongTensor)"""
        X_train, X_test, y_train, y_test = self.load_data()
        
        train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.long), torch.tensor(y_train, dtype=torch.long))
        test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.long), torch.tensor(y_test, dtype=torch.long))
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        
        return train_loader, test_loader, self.vocab_size

# --- 工具类 3: 通用训练器 ---
class ModelTrainer:
    def __init__(self, device=Config.DEVICE):
        self.device = device

    def _print_metrics(self, name, y_true, y_pred):
        """内部辅助函数：计算并打印核心指标"""
        acc = accuracy_score(y_true, y_pred)
        # binary模式：假设 1 (Botnet) 是正类，只关心正类的表现
        # 如果是多分类，需要改用 average='macro' 或 'weighted'
        p = precision_score(y_true, y_pred, average='binary', zero_division=0)
        r = recall_score(y_true, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
        
        print(f"[{name}] 测试结果:")
        print(f"  - Accuracy : {acc:.4f}")
        print(f"  - Precision: {p:.4f}")
        print(f"  - Recall   : {r:.4f}")
        print(f"  - F1-Score : {f1:.4f}")
        print("-" * 40)

    def train_sklearn(self, model, X_train, y_train, X_test, y_test, name="Model"):
        print(f"\n[{name}] 开始训练...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        # 打印详细指标
        self._print_metrics(name, y_test, preds)
        
        # print(classification_report(y_test, preds, target_names=['Benign', 'Botnet']))
        
        return model

    def train_pytorch(self, model, train_loader, test_loader, name="DL_Model", epochs=Config.EPOCHS, lr=Config.LR):
        print(f"\n[{name}] 开始训练 (Device: {self.device})...")
        model = model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # 动态判断输入类型
        input_dtype = torch.float32
        if hasattr(model, 'get_input_dtype'):
            input_dtype = model.get_input_dtype()

        # --- 训练循环 ---
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for X, y in train_loader:
                X, y = X.to(self.device).to(dtype=input_dtype), y.to(self.device)
                optimizer.zero_grad()
                out = model(X)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f}")

        # --- 评估循环 (修改部分) ---
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(self.device).to(dtype=input_dtype), y.to(self.device)
                out = model(X)
                _, pred = torch.max(out, 1)
                
                # 收集预测结果和真实标签，转回 CPU
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(y.cpu().numpy())
        
        # 统一计算指标
        self._print_metrics(name, all_targets, all_preds)
        
        return model
    