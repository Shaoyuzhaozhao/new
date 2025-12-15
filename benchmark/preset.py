"""
配置参数定义
"""

preset = {
    # 数据配置
    "data": {
        "path_data_x": "dataset/wifi_csi/mat",      # CSI幅度数据路径
        "path_data_mat": "dataset/wifi_csi/mat",    # 原始CSI数据路径
        "path_data_y": "dataset/annotation.csv",    # 标注文件路径
        "num_users": [ "1", "2", "3", "4", "5"],  # 使用所有用户数量,                          # 用户数量: ["0","1","2","3","4","5"]
        "wifi_band": ["5"],                        # WiFi频段: ["2.4"], ["5"], ["2.4", "5"]
        "environment": ["classroom"],  # 环境 , "meeting_room", "empty_room"
        "train_ratio": 0.8,                          # 训练集比例
        "num_workers": 4,                            # 数据加载线程数
    },
    
    # STFT配置
    "stft": {
        "n_fft": 256,                               # FFT点数
        "hop_length": 64,                           # 跳跃长度
        "win_length": 256,                          # 窗口长度
        "window": "hann",                           # 窗口类型
    },
    
    # 数据增强配置
    "augmentation": {
        "enabled": True,                            # 是否启用数据增强
        "fda": {                                    # 频域增强
            "enabled": True,
            "num_bands": 6,                         # 子带数量N
            "motion_threshold": 0.2,                # 运动阈值
        },
        "tda": {                                    # 时域增强
            "enabled": True,
            "scales": [0.5, 1.0, 2.0],             # 窗口尺度
            "base_window": 256,                     # 基准窗口大小
        },
        "mda": {                                    # 运动感知增强
            "enabled": True,
            "erase_ratio": 0.1,                     # 运动区域边缘擦除比例
            "static_erase_ratio": 0.5,              # 静态区域擦除比例
            "energy_threshold": 0.6,                # 能量阈值
        },
    },
    
    # 模型配置
    "model": {
        "name": "SDAN",                             # 模型名称
        "input_channels": 1,                        # 输入通道数
        "num_classes": 9,                           # 类别数
        "multi_scale_channels": [32, 32, 32, 32],   # 多尺度卷积通道数
        "kernel_sizes": [3, 5, 7],                  # 多尺度卷积核大小
        "temporal_kernel": 7,                       # 时间卷积核大小
        "frequency_kernel": 7,                      # 频率卷积核大小
        "attention_reduction": 16,                  # 注意力降维比例
        "dropout": 0.5,                             # Dropout比例
    },
    
    # 训练配置
    "train": {
        "batch_size": 128,
        "epochs": 100,
        "learning_rate": 1e-3,
        "min_lr": 1e-6,
        "weight_decay": 1e-4,
        "optimizer": "adam",                        # 优化器: adam, sgd
        "scheduler": "cosine",                      # 学习率调度: cosine, step, none
        "early_stopping": 1000,                       # 早停patience
    },
    
    # 评估配置
    "eval": {
        "metrics": ["accuracy", "precision", "recall", "f1"],
    },
    
    # 任务配置
    "task": {
        "name": "activity",                         # 任务: activity, identity, location
        "num_classes": {
            "activity": 9,                          # 9种活动
            "identity": 6,                          # 6个用户
            "location": 15,                         # 15个位置
        },
    },
    
    # 结果保存配置
    "output": {
        "path_result": "results/result.json",
        "path_model": "results/models",
        "path_log": "results/logs",
        "path_figure": "visualize",
    },
    
    # 设备配置
    "device": {
        "gpu": True,
        "gpu_id": 0,
    },
    
    # 随机种子
    "seed": 42,
    
    # 重复实验次数
    "repeat": 10,
}

# 活动类别映射
ACTIVITY_LABELS = {
    0: "No Action",
    1: "Walk",
    2: "Rotate",
    3: "Jump",
    4: "Wave",
    5: "Lie Down",
    6: "Pick Up",
    7: "Sit Down",
    8: "Stand Up",
}

# 环境映射
ENVIRONMENT_MAP = {
    "classroom": "Classroom",
    "meeting_room": "Meeting Room",
    "empty_room": "Empty Room",
}
