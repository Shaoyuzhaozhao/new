"""
CSI信号到时频频谱图的转换模块

实现论文3.1节的数据表示方法：
- 从一维CSI转换为二维时频频谱图
- 使用短时傅里叶变换(STFT)
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal
from typing import Optional, Tuple, Union


class SpectrogramTransform:
    """
    CSI到时频频谱图的转换类
    
    将CSI信号通过STFT变换为二维频谱图，保留时域和频域信息
    """
    
    def __init__(
        self,
        n_fft: int = 256,
        hop_length: int = 64,
        win_length: Optional[int] = None,
        window: str = 'hann',
        center: bool = True,
        pad_mode: str = 'reflect',
        normalized: bool = True,
        onesided: bool = True,
    ):
        """
        初始化频谱图变换器
        
        Args:
            n_fft: FFT点数，决定频率分辨率
            hop_length: 帧移长度，决定时间分辨率
            win_length: 窗口长度，默认等于n_fft
            window: 窗口类型 ('hann', 'hamming', 'blackman')
            center: 是否中心填充
            pad_mode: 填充模式
            normalized: 是否归一化
            onesided: 是否只返回单边频谱
        """
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length if win_length is not None else n_fft
        self.window = window
        self.center = center
        self.pad_mode = pad_mode
        self.normalized = normalized
        self.onesided = onesided
        
        # 创建窗口函数
        self.window_tensor = self._create_window()
    
    def _create_window(self) -> torch.Tensor:
        """创建窗口函数"""
        if self.window == 'hann':
            return torch.hann_window(self.win_length)
        elif self.window == 'hamming':
            return torch.hamming_window(self.win_length)
        elif self.window == 'blackman':
            return torch.blackman_window(self.win_length)
        else:
            return torch.ones(self.win_length)
    
    def transform(
        self, 
        csi: Union[np.ndarray, torch.Tensor],
        return_numpy: bool = False
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        将CSI信号转换为时频频谱图
        
        Args:
            csi: CSI信号，形状为 (T,) 或 (C, T) 或 (B, C, T)
            return_numpy: 是否返回numpy数组
            
        Returns:
            频谱图，形状为 (F, T') 或 (C, F, T') 或 (B, C, F, T')
        """
        # 转换为torch tensor
        if isinstance(csi, np.ndarray):
            csi = torch.from_numpy(csi).float()
        
        # 确保window在同一设备上
        if self.window_tensor.device != csi.device:
            self.window_tensor = self.window_tensor.to(csi.device)
        
        # 处理不同维度的输入
        original_ndim = csi.ndim
        if csi.ndim == 1:
            csi = csi.unsqueeze(0).unsqueeze(0)  # (1, 1, T)
        elif csi.ndim == 2:
            csi = csi.unsqueeze(0)  # (1, C, T)
        
        batch_size, num_channels, seq_len = csi.shape
        
        # 如果信号太短，进行填充
        min_length = self.n_fft
        if seq_len < min_length:
            pad_length = min_length - seq_len
            csi = torch.nn.functional.pad(csi, (0, pad_length), mode='constant', value=0)
            seq_len = min_length

        # 动态调整参数以适应信号长度
        n_fft = min(self.n_fft, seq_len)
        win_length = min(self.win_length, seq_len)
        hop_length = max(1, min(self.hop_length, seq_len // 4))

        # 确保n_fft >= win_length
        if n_fft < win_length:
            n_fft = win_length

        # 创建适当大小的窗口
        if win_length != self.win_length:
            if self.window == 'hann':
                window = torch.hann_window(win_length, device=csi.device)
            elif self.window == 'hamming':
                window = torch.hamming_window(win_length, device=csi.device)
            else:
                window = torch.ones(win_length, device=csi.device)
        else:
            window = self.window_tensor

        # 对每个通道进行STFT
        spectrograms = []
        for b in range(batch_size):
            channel_specs = []
            for c in range(num_channels):
                # STFT变换
                spec = torch.stft(
                    csi[b, c],
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=win_length,
                    window=window,
                    center=self.center,
                    pad_mode=self.pad_mode,
                    normalized=self.normalized,
                    onesided=self.onesided,
                    return_complex=True
                )
                # 取幅度谱
                spec_mag = torch.abs(spec)
                channel_specs.append(spec_mag)

            channel_specs = torch.stack(channel_specs, dim=0)  # (C, F, T')
            spectrograms.append(channel_specs)

        spectrograms = torch.stack(spectrograms, dim=0)  # (B, C, F, T')

        # 恢复原始维度
        if original_ndim == 1:
            spectrograms = spectrograms.squeeze(0).squeeze(0)  # (F, T')
        elif original_ndim == 2:
            spectrograms = spectrograms.squeeze(0)  # (C, F, T')

        if return_numpy:
            return spectrograms.numpy()
        return spectrograms

    def transform_multiscale(
        self,
        csi: Union[np.ndarray, torch.Tensor],
        scales: list = [0.5, 1.0, 2.0],
        return_numpy: bool = False
    ) -> list:
        """
        多尺度时间窗口的STFT变换 (用于TDA增强)

        Args:
            csi: CSI信号
            scales: 窗口尺度列表
            return_numpy: 是否返回numpy数组

        Returns:
            不同尺度的频谱图列表
        """
        spectrograms = []
        base_win_length = self.win_length

        for scale in scales:
            # 调整窗口长度
            scaled_win_length = int(base_win_length * scale)
            # 确保是偶数
            scaled_win_length = scaled_win_length if scaled_win_length % 2 == 0 else scaled_win_length + 1
            scaled_n_fft = max(scaled_win_length, self.n_fft)

            # 创建新的窗口
            if self.window == 'hann':
                window = torch.hann_window(scaled_win_length)
            elif self.window == 'hamming':
                window = torch.hamming_window(scaled_win_length)
            else:
                window = torch.ones(scaled_win_length)

            # 转换为tensor
            if isinstance(csi, np.ndarray):
                csi_tensor = torch.from_numpy(csi).float()
            else:
                csi_tensor = csi.float()

            if window.device != csi_tensor.device:
                window = window.to(csi_tensor.device)

            # STFT变换
            if csi_tensor.ndim == 1:
                spec = torch.stft(
                    csi_tensor,
                    n_fft=scaled_n_fft,
                    hop_length=self.hop_length,
                    win_length=scaled_win_length,
                    window=window,
                    center=self.center,
                    pad_mode=self.pad_mode,
                    normalized=self.normalized,
                    onesided=self.onesided,
                    return_complex=True
                )
                spec_mag = torch.abs(spec)
            else:
                # 处理多维输入
                spec_mag = self.transform(csi_tensor, return_numpy=False)

            if return_numpy:
                spectrograms.append(spec_mag.numpy())
            else:
                spectrograms.append(spec_mag)

        return spectrograms


class CSIToSpectrogram:
    """
    完整的CSI到频谱图转换管道

    包括：
    1. CSI幅度/相位提取
    2. STFT变换
    3. 频谱图归一化
    """

    def __init__(
        self,
        n_fft: int = 256,
        hop_length: int = 64,
        normalize: bool = True,
        log_scale: bool = True,
        use_amplitude: bool = True,
    ):
        """
        初始化转换管道

        Args:
            n_fft: FFT点数
            hop_length: 帧移长度
            normalize: 是否归一化
            log_scale: 是否使用对数刻度
            use_amplitude: 是否使用幅度(True)或使用复数(False)
        """
        self.stft_transform = SpectrogramTransform(
            n_fft=n_fft,
            hop_length=hop_length,
        )
        self.normalize = normalize
        self.log_scale = log_scale
        self.use_amplitude = use_amplitude

    def __call__(
        self,
        csi_data: Union[np.ndarray, torch.Tensor],
        subcarrier_idx: Optional[int] = None,
        link_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """
        将CSI数据转换为频谱图

        Args:
            csi_data: CSI数据，形状可以是:
                - (T, Nt, Nr, K): 完整CSI张量
                - (T, num_links, K): 简化的CSI张量
                - (T, K): 单链路CSI
                - (T,): 单子载波CSI
            subcarrier_idx: 指定子载波索引
            link_idx: 指定链路索引

        Returns:
            频谱图tensor
        """
        # 转换为numpy进行处理
        if isinstance(csi_data, torch.Tensor):
            csi_data = csi_data.numpy()

        # 处理不同维度的CSI数据
        if csi_data.ndim == 4:
            # (T, Nt, Nr, K) -> 选择特定链路和子载波或平均
            T, Nt, Nr, K = csi_data.shape
            if link_idx is not None and subcarrier_idx is not None:
                nt_idx = link_idx // Nr
                nr_idx = link_idx % Nr
                csi_signal = csi_data[:, nt_idx, nr_idx, subcarrier_idx]
            elif subcarrier_idx is not None:
                # 平均所有链路
                csi_signal = csi_data[:, :, :, subcarrier_idx].mean(axis=(1, 2))
            else:
                # 平均所有链路和子载波
                csi_signal = csi_data.mean(axis=(1, 2, 3))
        elif csi_data.ndim == 3:
            # (T, num_links, K)
            if subcarrier_idx is not None:
                csi_signal = csi_data[:, :, subcarrier_idx].mean(axis=1)
            else:
                csi_signal = csi_data.mean(axis=(1, 2))
        elif csi_data.ndim == 2:
            # (T, K)
            if subcarrier_idx is not None:
                csi_signal = csi_data[:, subcarrier_idx]
            else:
                csi_signal = csi_data.mean(axis=1)
        else:
            # (T,)
            csi_signal = csi_data

        # STFT变换
        spectrogram = self.stft_transform.transform(csi_signal, return_numpy=False)

        # 对数刻度
        if self.log_scale:
            spectrogram = torch.log1p(spectrogram)

        # 归一化
        if self.normalize:
            mean = spectrogram.mean()
            std = spectrogram.std()
            if std > 0:
                spectrogram = (spectrogram - mean) / std

        return spectrogram


def compute_motion_statistics(csi_signal: np.ndarray) -> float:
    """
    计算CSI信号的运动统计量

    论文公式: MS[k] = Σt|h[k,t]·h*[k,t-1]| / Σt|h[k,t]|²

    Args:
        csi_signal: CSI时间序列，形状为 (T,)

    Returns:
        运动统计量，值越低表示运动越强
    """
    if len(csi_signal) < 2:
        return 1.0

    # 计算自相关的首个非零滞后值
    numerator = np.sum(np.abs(csi_signal[1:] * np.conj(csi_signal[:-1])))
    denominator = np.sum(np.abs(csi_signal) ** 2)

    if denominator == 0:
        return 1.0

    return numerator / denominator


def select_motion_sensitive_subcarriers(
    csi_data: np.ndarray,
    num_bands: int = 6,
) -> list:
    """
    智能子载波选择策略 (ISS-N)

    将子载波分为N个子带，在每个子带内选择运动敏感度最高的子载波

    Args:
        csi_data: CSI数据，形状可以是 (T,), (T, K), (T, num_links, K), 或 (T, Nt, Nr, K)
        num_bands: 子带数量N

    Returns:
        选中的子载波索引列表
    """
    # 处理不同维度的输入
    if csi_data.ndim == 1:
        # (T,) - 单子载波，直接返回索引0
        return [0] * num_bands
    elif csi_data.ndim == 2:
        # (T, K) - 已经是正确格式
        pass
    elif csi_data.ndim == 3:
        # (T, num_links, K) - 对链路维度取平均
        csi_data = csi_data.mean(axis=1)
    elif csi_data.ndim == 4:
        # (T, Nt, Nr, K) - 对天线维度取平均
        csi_data = csi_data.mean(axis=(1, 2))
    else:
        # 其他情况，尝试展平到2D
        original_shape = csi_data.shape
        T = original_shape[0]
        csi_data = csi_data.reshape(T, -1)

    T, K = csi_data.shape

    # 如果子载波数少于子带数，调整子带数
    if K < num_bands:
        num_bands = max(1, K)

    # 计算每个子载波的运动统计量
    motion_stats = []
    for k in range(K):
        ms = compute_motion_statistics(csi_data[:, k])
        motion_stats.append(ms)
    motion_stats = np.array(motion_stats)

    # 运动敏感度 = 1 - MS
    motion_sensitivity = 1 - motion_stats

    # 划分子带
    subcarriers_per_band = K // num_bands
    if subcarriers_per_band == 0:
        subcarriers_per_band = 1

    selected_subcarriers = []

    for i in range(num_bands):
        start_idx = i * subcarriers_per_band
        end_idx = start_idx + subcarriers_per_band if i < num_bands - 1 else K

        if start_idx >= K:
            break

        # 在子带内选择运动敏感度最高的子载波
        band_sensitivity = motion_sensitivity[start_idx:end_idx]
        best_idx = start_idx + np.argmax(band_sensitivity)
        selected_subcarriers.append(best_idx)

    return selected_subcarriers


def detect_motion_regions(
    spectrogram: np.ndarray,
    energy_threshold: float = 0.7,
) -> Tuple[list, list]:
    """
    检测频谱图中的运动区域

    Args:
        spectrogram: 频谱图，形状为 (F, T), (C, F, T), 或更高维
        energy_threshold: 能量阈值

    Returns:
        (motion_frames, static_frames): 运动帧索引列表和静态帧索引列表
    """
    # 处理不同维度的输入，确保是2D (F, T)
    if spectrogram.ndim == 1:
        # (T,) -> 无法检测，返回全部为运动帧
        T = spectrogram.shape[0]
        return list(range(T)), []
    elif spectrogram.ndim == 2:
        # (F, T) - 正确格式
        pass
    elif spectrogram.ndim == 3:
        # (C, F, T) 或 (T, F, bins) - 取平均或选择
        if spectrogram.shape[0] < spectrogram.shape[1]:
            # 可能是 (C, F, T)，对通道取平均
            spectrogram = spectrogram.mean(axis=0)
        else:
            # 可能是 (T, F, bins)，取最后一个维度的平均
            spectrogram = spectrogram.mean(axis=-1).T  # 转为 (F, T)
    else:
        # 更高维度，展平到2D
        shape = spectrogram.shape
        spectrogram = spectrogram.reshape(-1, shape[-1])

    F, T = spectrogram.shape

    # 计算每个时间帧的能量集中度
    energy_concentration = []
    for t in range(T):
        frame = spectrogram[:, t]
        # E_τ = Σf|S(τ,f)|² / (Σf|S(τ,f)|)²
        sum_sq = np.sum(frame ** 2)
        sq_sum = np.sum(frame) ** 2
        if sq_sum > 0:
            ec = sum_sq / sq_sum
        else:
            ec = 0
        energy_concentration.append(ec)

    energy_concentration = np.array(energy_concentration)

    # 归一化
    if energy_concentration.max() > energy_concentration.min():
        ec_norm = (energy_concentration - energy_concentration.min()) / \
                  (energy_concentration.max() - energy_concentration.min())
    else:
        ec_norm = energy_concentration

    # 分类运动帧和静态帧
    motion_frames = np.where(ec_norm > energy_threshold)[0].tolist()
    static_frames = np.where(ec_norm <= energy_threshold)[0].tolist()

    return motion_frames, static_frames