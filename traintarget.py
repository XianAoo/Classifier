import sys
import os
import torch
import joblib

# 引用本地模块
from train_utils import FlowDataManager, SequenceDataManager, ModelTrainer, Config
from TargetModel import (
    RandomForestModel, XGBoostModel, 
    MLP, AlertNet, DeepNet, IdsNet, 
    FSNet, MaMPF_LengthOnly
)

# 定义模型保存路径
MODELS_DIR = 'targetmodels'

def run_training_pipeline():
    # 0. 确保保存目录存在
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        print(f"创建模型保存目录: {MODELS_DIR}")

    # 初始化通用训练器
    trainer = ModelTrainer()
    
    print("="*40)
    print("阶段一：基于流量统计特征 (CSV) 的模型训练")
    print("="*40)
    
    # 1. 准备 CSV 数据
    flow_manager = FlowDataManager()
    
    # 1.1 获取 Numpy 数据 (给 sklearn 用)
    X_train_np, X_test_np, y_train_np, y_test_np = flow_manager.load_data()
    
    # 1.2 获取 PyTorch DataLoaders (给 深度学习 用)
    flow_train_loader, flow_test_loader, input_dim = flow_manager.get_dataloaders()
    print(f"统计特征维度: {input_dim}")

    # --- 训练并保存 sklearn 模型 ---
    # 1. RandomForest
    rf_model = RandomForestModel()
    trainer.train_sklearn(
        rf_model, X_train_np, y_train_np, X_test_np, y_test_np, name="RandomForest"
    )
    # 保存 (RandomForestModel 类内部封装了 sklearn 对象，这里直接保存整个 wrapper 类)
    joblib.dump(rf_model, os.path.join(MODELS_DIR, 'RandomForest.pkl'))
    print(f"  -> 模型已保存: RandomForest.pkl")
    
    # 2. XGBoost
    xgb_model = XGBoostModel()
    trainer.train_sklearn(
        xgb_model, X_train_np, y_train_np, X_test_np, y_test_np, name="XGBoost"
    )
    joblib.dump(xgb_model, os.path.join(MODELS_DIR, 'XGBoost.pkl'))
    print(f"  -> 模型已保存: XGBoost.pkl")

    # --- 训练并保存 PyTorch 深度学习模型 ---
    # 定义要训练的模型清单
    flow_dl_models = [
        ("MLP",      MLP(input_size=input_dim, output_size=2)),
        ("AlertNet", AlertNet(input_dim=input_dim, num_classes=2)),
        ("DeepNet",  DeepNet(input_dim=input_dim, num_classes=2)),
        ("IdsNet",   IdsNet(input_dim=input_dim, num_classes=2, dataset_type='Custom'))
    ]

    for name, model in flow_dl_models:
        # 训练
        trained_model = trainer.train_pytorch(model, flow_train_loader, flow_test_loader, name=name)
        
        # 保存 (推荐保存 state_dict 权重，而不是整个模型对象)
        save_path = os.path.join(MODELS_DIR, f'{name}.pth')
        torch.save(trained_model.state_dict(), save_path)
        print(f"  -> 模型权重已保存: {name}.pth")


    print("\n" + "="*40)
    print("阶段二：基于序列特征 (JSON) 的模型训练")
    print("="*40)

    # 2. 准备序列数据
    seq_manager = SequenceDataManager()
    
    # 2.1 获取 Numpy 数据 (给 MaMPF 用)
    X_seq_train, X_seq_test, y_seq_train, y_seq_test = seq_manager.load_data()
    
    # 2.2 获取 PyTorch DataLoaders (给 FSNet 用)
    seq_train_loader, seq_test_loader, vocab_size = seq_manager.get_dataloaders()
    print(f"序列最大词汇量 (Vocab Size): {vocab_size}")

    # --- 训练并保存 MaMPF (统计方法) ---
    mampf = MaMPF_LengthOnly()
    trainer.train_sklearn(
        mampf, X_seq_train, y_seq_train, X_seq_test, y_seq_test, name="MaMPF"
    )
    joblib.dump(mampf, os.path.join(MODELS_DIR, 'MaMPF.pkl'))
    print(f"  -> 模型已保存: MaMPF.pkl")

    # --- 训练并保存 FSNet (深度学习) ---
    fsnet = FSNet(num_classes=2, max_len=Config.SEQ_LEN, vocab_size=vocab_size)
    
    trained_fsnet = trainer.train_pytorch(fsnet, seq_train_loader, seq_test_loader, name="FSNet")
    
    # 保存
    torch.save(trained_fsnet.state_dict(), os.path.join(MODELS_DIR, 'FSNet.pth'))
    print(f"  -> 模型权重已保存: FSNet.pth")

    print(f"\n所有模型训练完成！文件均保存在 '{MODELS_DIR}/' 目录下。")

if __name__ == "__main__":
    run_training_pipeline()