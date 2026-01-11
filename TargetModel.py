import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost
from sklearn.ensemble import RandomForestClassifier

import numpy as np
from collections import Counter, defaultdict
from sklearn.base import BaseEstimator, ClassifierMixin

# --- 1. 定义抽象基类 (统一接口) ---
class AbstractDLModel(nn.Module):
    """
    深度学习模型的统一基类。
    规范：
    1. forward 输出必须是 logits (不带 sigmoid/softmax)。
    2. 提供 get_input_dtype 方法告诉 Trainer 需要什么类型的数据。
    """
    def __init__(self):
        super(AbstractDLModel, self).__init__()

    def get_input_dtype(self):
        """默认模型都需要 Float 类型的特征向量"""
        return torch.float32

# --- 2. 经典机器学习模型 (保持不变) ---
class RandomForestModel:
    def __init__(self, n_trees=10, split_criterion='gini', max_depth=None, min_samples_leaf=1, min_samples_split=2, random_state=0):
        self.model = RandomForestClassifier(n_estimators=n_trees, criterion=split_criterion, max_depth=max_depth, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split, random_state=random_state)
    def fit(self, X, y): self.model.fit(X, y)
    def predict(self, X): return self.model.predict(X)

class XGBoostModel:
    def __init__(self, input_size=None, output_size=None):
        self.model = xgboost.XGBClassifier()
    def fit(self, X, y): self.model.fit(X, y)
    def predict(self, X): return self.model.predict(X)

# --- 3. 深度学习模型 (统一修改) ---
class MLP(AbstractDLModel):
    """
    [修改] 移除 Sigmoid，增加 Dropout 参数
    """
    def __init__(self, input_size, output_size, dropout_rate=0.0):
        super(MLP, self).__init__()
        self.fc_1 = nn.Linear(input_size, 32)
        self.fc_2 = nn.Linear(32, 10)
        self.fc_3 = nn.Linear(10, output_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.fc_1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc_2(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc_3(x)
        return x  # 返回 logits

class AlertNet(AbstractDLModel):
    """
    [修改] 继承 AbstractDLModel，参数化 Dropout
    """
    def __init__(self, input_dim, num_classes, dropout_rate=0.01):
        super(AlertNet, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(input_dim, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(p=dropout_rate),
            nn.Linear(1024, 768), nn.BatchNorm1d(768), nn.ReLU(), nn.Dropout(p=dropout_rate),
            nn.Linear(768, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(p=dropout_rate),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(p=dropout_rate)
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class DeepNet(AbstractDLModel):
    def __init__(self, input_dim, num_classes, dropout_rate=0.01):
        super(DeepNet, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(p=dropout_rate),
            nn.Linear(256, 256), nn.ReLU(), nn.Dropout(p=dropout_rate),
            nn.Linear(256, 256), nn.ReLU(), nn.Dropout(p=dropout_rate),
            nn.Linear(256, 256), nn.ReLU(), nn.Dropout(p=dropout_rate)
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class IdsNet(AbstractDLModel):
    def __init__(self, input_dim, num_classes, dataset_type='CICIDS-2017', dropout_rate=0.01):
        super(IdsNet, self).__init__()
        if dataset_type == 'CICIDS-2017': h1, h2 = 42, 21
        elif dataset_type == 'NSL-KDD': h1, h2 = 64, 32
        else: h1, h2 = 64, 32

        self.features = nn.Sequential(
            nn.Linear(input_dim, h1), nn.ReLU(), nn.Dropout(p=dropout_rate),
            nn.Linear(h1, h2), nn.ReLU(), nn.Dropout(p=dropout_rate)
        )
        self.classifier = nn.Linear(h2, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class FSNet(AbstractDLModel):
    """
    FS-Net (无 Decoder 版 / FS-ND)
    """
    def __init__(self, num_classes, max_len, vocab_size, embed_dim=128, hidden_dim=128, num_layers=2, dropout_rate=0.3):
        super(FSNet, self).__init__()
        self.max_len = max_len
        
        # 1. Embedding
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
        
        # 2. Encoder (Bi-GRU)
        self.encoder = nn.GRU(
            input_size=embed_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=True,
            dropout=dropout_rate
        )
        
        # Encoder 输出特征维度 = 层数 * 2 (双向) * 隐藏层维度
        # 因为没有 Decoder，不需要拼接 z_d，也不需要 feature fusion
        self.feature_dim = num_layers * 2 * hidden_dim 
        
        # 3. Dense Layer (压缩特征)
        # 输入维度直接是 Encoder 的输出特征维度
        self.dense = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU()
        )
        
        # 4. Classifier
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def get_input_dtype(self):
        """FSNet 需要 Long 类型的整数索引作为输入"""
        return torch.long

    def forward(self, x):
        # 鲁棒性检查：确保输入是 Long 类型
        if x.dtype != torch.long:
            x = x.long()
            
        # (Batch, Seq_Len) -> (Batch, Seq_Len, Embed_Dim)
        emb = self.embedding(x) 
        
        # Encoder 前向传播
        # out: (Batch, Seq_Len, Hidden*2)
        # h_n: (Layers*2, Batch, Hidden)
        _, h_n = self.encoder(emb)
        
        # [cite_start]提取特征 z_e [cite: 201]
        # 将 h_n 变换为 (Batch, Layers*2*Hidden)
        batch_size = x.size(0)
        z_e = h_n.permute(1, 0, 2).reshape(batch_size, -1)
        
        # 直接通过 Dense 层和分类层 (无 Fusion)
        z_c = self.dense(z_e)
        logits = self.classifier(z_c)
        
        return logits

# --- 4. MaMPF ---
class MaMPF_LengthOnly(BaseEstimator, ClassifierMixin):
    def __init__(self, coverage_ratio=0.90, n_estimators=100, smoothing=1e-6):
        self.coverage_ratio = coverage_ratio
        self.n_estimators = n_estimators
        self.smoothing = smoothing
        self.classes_ = None
        self.length_blocks_ = {} 
        self.markov_models_ = {}
        self.final_classifier = RandomForestClassifier(n_estimators=n_estimators, random_state=42)

    def _get_length_blocks(self, sequences):
        total_count = 0
        counts = Counter()
        for seq in sequences:
            counts.update(seq)
            total_count += len(seq)
        sorted_lengths = counts.most_common()
        current_cover = 0
        blocks = []
        target_count = total_count * self.coverage_ratio
        for length, count in sorted_lengths:
            blocks.append(length)
            current_cover += count
            if current_cover >= target_count:
                break
        return np.array(blocks)

    def _map_sequence_to_blocks(self, sequence, blocks):
        if len(blocks) == 0: return sequence 
        seq_arr = np.array(sequence).reshape(-1, 1)
        dist = np.abs(seq_arr - blocks)
        min_idx = np.argmin(dist, axis=1)
        return blocks[min_idx]

    def _train_markov_chain(self, sequences, blocks):
        transitions = defaultdict(lambda: defaultdict(int))
        for seq in sequences:
            mapped_seq = self._map_sequence_to_blocks(seq, blocks)
            for t in range(len(mapped_seq) - 1):
                transitions[mapped_seq[t]][mapped_seq[t+1]] += 1
        trans_prob = defaultdict(dict)
        for state, next_states in transitions.items():
            total = sum(next_states.values())
            for next_s, count in next_states.items():
                trans_prob[state][next_s] = count / total
        return trans_prob

    def _calculate_score(self, sequence, blocks, trans_prob):
        if len(sequence) < 2: return self.smoothing
        mapped_seq = self._map_sequence_to_blocks(sequence, blocks)
        log_prob = 0.0
        hits = 0
        for t in range(len(mapped_seq) - 1):
            p = trans_prob.get(mapped_seq[t], {}).get(mapped_seq[t+1], self.smoothing)
            if p == 0: p = self.smoothing
            log_prob += np.log(p)
            hits += 1
        if hits == 0: return self.smoothing
        return np.exp(log_prob / hits)

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        X_by_class = defaultdict(list)
        for seq, label in zip(X, y):
            X_by_class[label].append(seq)
        for label in self.classes_:
            seqs = X_by_class[label]
            blocks = self._get_length_blocks(seqs)
            self.length_blocks_[label] = blocks
            self.markov_models_[label] = self._train_markov_chain(seqs, blocks)
        X_features = []
        for seq in X:
            feature_vec = [self._calculate_score(seq, self.length_blocks_[l], self.markov_models_[l]) for l in self.classes_]
            X_features.append(feature_vec)
        self.final_classifier.fit(X_features, y)
        return self

    def predict(self, X):
        X_features = []
        for seq in X:
            feature_vec = [self._calculate_score(seq, self.length_blocks_[l], self.markov_models_[l]) for l in self.classes_]
            X_features.append(feature_vec)
        return self.final_classifier.predict(X_features)
    
