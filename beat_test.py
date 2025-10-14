import os
import argparse
import json
from typing import Tuple, List, Dict, Optional

import numpy as np


def _import_madmom_or_raise():
    try:
        # Lazy import to avoid hard dependency when this file is just imported
        from madmom.features.beats import (
            RNNBeatProcessor,
            DBNBeatTrackingProcessor,
        )
        return RNNBeatProcessor, DBNBeatTrackingProcessor
    except Exception as exc:
        raise ImportError(
            "需要安装 madmom 才能进行音频节拍检测，请先安装：pip install madmom"
        ) from exc


def detect_audio_beats(audio_path: str, dbn_fps: int = 100) -> np.ndarray:
    """使用 madmom 从音频中提取节拍时间（秒）.

    Args:
        audio_path: 音频文件路径（如 .wav）
        dbn_fps: DBN 跟踪处理器的帧率，用于时间分辨率

    Returns:
        1D numpy 数组，单位为秒的节拍时间点，升序排序
    """
    RNNBeatProcessor, DBNBeatTrackingProcessor = _import_madmom_or_raise()

    beat_act_processor = RNNBeatProcessor()
    beat_activations = beat_act_processor(audio_path)
    dbn_processor = DBNBeatTrackingProcessor(fps=dbn_fps)
    beat_times: np.ndarray = dbn_processor(beat_activations)
    if beat_times.ndim != 1:
        beat_times = np.asarray(beat_times).reshape(-1)
    return np.sort(beat_times.astype(float))


def get_audio_beat_activations(audio_path: str) -> Tuple[np.ndarray, float]:
    """获取 madmom RNN 的节拍激活序列与其帧率（通常为 100Hz）."""
    RNNBeatProcessor, _ = _import_madmom_or_raise()
    beat_act_processor = RNNBeatProcessor()
    beat_activations = beat_act_processor(audio_path)
    act_fps = 100.0  # RNNBeatProcessor 默认 100 Hz
    return np.asarray(beat_activations, dtype=float).reshape(-1), act_fps


def _load_motion_array(motion_path: str) -> np.ndarray:
    """加载动作数据为 (num_frames, num_features) 的 numpy 数组.

    - 支持 .npy（推荐）和 .csv（逗号分隔）
    """
    ext = os.path.splitext(motion_path)[1].lower()
    if ext == ".npy":
        arr = np.load(motion_path)
    elif ext in {".csv", ".txt"}:
        arr = np.loadtxt(motion_path, delimiter=",")
    else:
        raise ValueError(f"不支持的动作数据格式: {ext}, 仅支持 .npy / .csv / .txt")

    if arr.ndim != 2:
        raise ValueError(
            f"动作数组维度应为2维 (num_frames, num_features)，当前为 {arr.shape}"
        )
    return arr.astype(np.float32)


def compute_motion_energy_series(motion_features: np.ndarray) -> np.ndarray:
    """计算逐帧动作能量序列（用于检测动作节奏峰值）.

    策略：
    - 计算相邻帧的一阶差分
    - 在特征维度做 L2 范数（得到每帧的整体“速度/能量”）
    - 在起始帧前补 0 以与原帧数对齐
    """
    if motion_features.ndim != 2:
        raise ValueError("motion_features 应为二维数组 [num_frames, num_features]")

    # 一阶差分（丢掉首帧），形状: (num_frames-1, num_features)
    frame_deltas = np.diff(motion_features, axis=0)
    per_frame_energy = np.linalg.norm(frame_deltas, axis=1)
    # 与原序列对齐（首帧无差分，补0）
    energy_aligned = np.concatenate([np.zeros(1, dtype=per_frame_energy.dtype), per_frame_energy])
    return energy_aligned


def detect_motion_peaks(
    energy_series: np.ndarray,
    fps_motion: float,
    min_interval_s: float = 0.2,
    prominence_scale: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """从动作能量序列中检测峰值所在的时间点（秒）.

    Args:
        energy_series: 逐帧能量序列
        fps_motion: 动作的帧率（帧/秒）
        min_interval_s: 相邻峰的最小时间间隔（秒）
        prominence_scale: 峰值显著性阈值 = prominence_scale * std(energy)

    Returns:
        (peak_times_sec, peak_indices)
    """
    from scipy.signal import find_peaks

    if energy_series.ndim != 1:
        raise ValueError("energy_series 必须为一维数组")

    std_energy = float(np.std(energy_series))
    # 自适应显著性阈值；若序列近似常数，给一个很小的阈值避免检出为空
    prominence = prominence_scale * std_energy if std_energy > 1e-8 else 1e-6
    min_distance_frames = max(1, int(round(min_interval_s * fps_motion)))

    peak_indices, _ = find_peaks(energy_series, distance=min_distance_frames, prominence=prominence)
    peak_times_sec = peak_indices.astype(float) / float(fps_motion)
    return np.asarray(peak_times_sec, dtype=float), np.asarray(peak_indices, dtype=int)


def detect_motion_valleys(
    energy_series: np.ndarray,
    fps_motion: float,
    min_interval_s: float = 0.2,
    prominence_scale: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """检测动作能量序列中的谷值（极小值）.

    通过对 -energy 使用同样的峰值检测策略来实现谷值检测。
    Returns: (valley_times_sec, valley_indices)
    """
    from scipy.signal import find_peaks

    if energy_series.ndim != 1:
        raise ValueError("energy_series 必须为一维数组")

    std_energy = float(np.std(energy_series))
    prominence = prominence_scale * std_energy if std_energy > 1e-8 else 1e-6
    min_distance_frames = max(1, int(round(min_interval_s * fps_motion)))

    valley_indices, _ = find_peaks(-energy_series, distance=min_distance_frames, prominence=prominence)
    valley_times_sec = valley_indices.astype(float) / float(fps_motion)
    return np.asarray(valley_times_sec, dtype=float), np.asarray(valley_indices, dtype=int)


def match_events(
    ref_times: np.ndarray,
    cand_times: np.ndarray,
    tolerance_s: float = 0.15,
) -> Dict[str, np.ndarray]:
    """贪心最近邻匹配事件时间序列.

    - 对于每个 ref（音频节拍），寻找最近且未匹配的 cand（动作峰）
    - 若时间差 <= 容差，计为匹配

    Returns:
        {
          'matched_ref_idx': [...],
          'matched_cand_idx': [...],
          'abs_errors_s': [...],
          'unmatched_ref_idx': [...],
          'unmatched_cand_idx': [...],
        }
    """
    ref_times = np.asarray(ref_times, dtype=float)
    cand_times = np.asarray(cand_times, dtype=float)
    if ref_times.size == 0 or cand_times.size == 0:
        return {
            "matched_ref_idx": np.array([], dtype=int),
            "matched_cand_idx": np.array([], dtype=int),
            "abs_errors_s": np.array([], dtype=float),
            "unmatched_ref_idx": np.arange(ref_times.size, dtype=int),
            "unmatched_cand_idx": np.arange(cand_times.size, dtype=int),
        }

    matched_ref_idx: List[int] = []
    matched_cand_idx: List[int] = []
    abs_errors: List[float] = []

    # 双指针贪心匹配（因两序列为升序）
    i, j = 0, 0
    while i < len(ref_times) and j < len(cand_times):
        dt = cand_times[j] - ref_times[i]
        if abs(dt) <= tolerance_s:
            matched_ref_idx.append(i)
            matched_cand_idx.append(j)
            abs_errors.append(abs(dt))
            i += 1
            j += 1
        elif dt < -tolerance_s:
            # cand 在 ref 之前太多，前进 cand
            j += 1
        else:
            # ref 在 cand 之前太多，前进 ref
            i += 1

    unmatched_ref = np.setdiff1d(np.arange(len(ref_times)), np.asarray(matched_ref_idx, dtype=int), assume_unique=False)
    unmatched_cand = np.setdiff1d(np.arange(len(cand_times)), np.asarray(matched_cand_idx, dtype=int), assume_unique=False)

    return {
        "matched_ref_idx": np.asarray(matched_ref_idx, dtype=int),
        "matched_cand_idx": np.asarray(matched_cand_idx, dtype=int),
        "abs_errors_s": np.asarray(abs_errors, dtype=float),
        "unmatched_ref_idx": unmatched_ref,
        "unmatched_cand_idx": unmatched_cand,
    }


def compute_alignment_metrics(
    audio_beats_s: np.ndarray,
    motion_peaks_s: np.ndarray,
    tolerance_s: float,
) -> Dict[str, float]:
    """根据匹配结果计算对齐指标（Precision/Recall/F1/MAE/coverage）.
    - Precision: TP / (TP + FP)  (动作峰有多少命中音频节拍)
    - Recall:    TP / (TP + FN)  (音频节拍有多少被动作峰命中)
    - F1:        2PR / (P + R)
    - MAE:       匹配对的平均绝对时间误差（秒）
    - BeatCoverage:   TP / #beats
    - MotionCoverage: TP / #peaks
    """
    match = match_events(audio_beats_s, motion_peaks_s, tolerance_s)
    tp = int(match["matched_ref_idx"].size)
    fn = int(match["unmatched_ref_idx"].size)
    fp = int(match["unmatched_cand_idx"].size)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    mae = float(np.mean(match["abs_errors_s"])) if tp > 0 else float("inf")

    return {
        "num_beats": int(len(audio_beats_s)),
        "num_motion_peaks": int(len(motion_peaks_s)),
        "tolerance_s": float(tolerance_s),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mae_s": float(mae),
    }


def run_metric(
    audio_path: str,
    motion_path: str,
    fps_motion: float = 30.0,
    tolerance_s: float = 0.15,
    min_interval_s: float = 0.2,
    prominence_scale: float = 0.5,
    dbn_fps: int = 100,
    save_debug_dir: Optional[str] = None,
    plot_path: Optional[str] = None,
) -> Dict[str, object]:
    """对齐音频节拍与动作节奏峰，输出指标与中间结果.

    Returns:
        dict 包含 metrics 与 beats/peaks 时间数组
    """
    audio_beats_s = detect_audio_beats(audio_path, dbn_fps=dbn_fps)
    motion_arr = _load_motion_array(motion_path)
    energy = compute_motion_energy_series(motion_arr)
    motion_peaks_s, motion_peak_idx = detect_motion_peaks(
        energy_series=energy,
        fps_motion=fps_motion,
        min_interval_s=min_interval_s,
        prominence_scale=prominence_scale,
    )
    motion_valleys_s, motion_valley_idx = detect_motion_valleys(
        energy_series=energy,
        fps_motion=fps_motion,
        min_interval_s=min_interval_s,
        prominence_scale=prominence_scale,
    )

    metrics = compute_alignment_metrics(audio_beats_s, motion_peaks_s, tolerance_s)

    result: Dict[str, object] = {
        "metrics": metrics,
        "audio_beats_s": audio_beats_s.tolist(),
        "motion_peaks_s": motion_peaks_s.tolist(),
        "motion_peak_indices": motion_peak_idx.tolist(),
        "motion_valleys_s": motion_valleys_s.tolist(),
        "motion_valley_indices": motion_valley_idx.tolist(),
    }

    if save_debug_dir:
        os.makedirs(save_debug_dir, exist_ok=True)
        # 保存 beats 与 peaks 为 csv 以便可视化
        np.savetxt(os.path.join(save_debug_dir, "audio_beats_s.csv"), audio_beats_s, delimiter=",", fmt="%.6f")
        np.savetxt(os.path.join(save_debug_dir, "motion_peaks_s.csv"), motion_peaks_s, delimiter=",", fmt="%.6f")
        np.savetxt(os.path.join(save_debug_dir, "motion_valleys_s.csv"), motion_valleys_s, delimiter=",", fmt="%.6f")
        np.savetxt(os.path.join(save_debug_dir, "motion_energy.csv"), energy, delimiter=",", fmt="%.8f")
        with open(os.path.join(save_debug_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

    # 绘图（若指定）
    final_plot_path: Optional[str] = plot_path
    if final_plot_path is None and save_debug_dir:
        final_plot_path = os.path.join(save_debug_dir, "beat_motion_rhythm.png")
    if final_plot_path is not None:
        try:
            # 尝试读取音频激活用于上半部分绘制
            beat_activations, act_fps = get_audio_beat_activations(audio_path)
        except Exception:
            beat_activations, act_fps = None, None
        _plot_rhythm(
            plot_path=final_plot_path,
            audio_beats_s=audio_beats_s,
            beat_activations=beat_activations,
            beat_activation_fps=act_fps,
            motion_energy=energy,
            fps_motion=fps_motion,
            motion_peak_idx=motion_peak_idx,
            motion_valley_idx=motion_valley_idx,
        )

    return result


def _plot_rhythm(
    plot_path: str,
    audio_beats_s: np.ndarray,
    beat_activations: Optional[np.ndarray],
    beat_activation_fps: Optional[float],
    motion_energy: np.ndarray,
    fps_motion: float,
    motion_peak_idx: np.ndarray,
    motion_valley_idx: np.ndarray,
) -> None:
    """绘制音频节拍与动作能量的节奏峰谷图并保存到文件."""
    import matplotlib
    matplotlib.use("Agg")  # 非交互式后端，便于服务器保存
    import matplotlib.pyplot as plt

    # 时间轴
    t_motion = np.arange(len(motion_energy), dtype=float) / float(fps_motion)

    # 布局：上（音频激活+beats），下（动作能量+峰谷+beats）
    nrows = 2 if beat_activations is not None and beat_activation_fps else 1
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(12, 6 if nrows == 1 else 8), sharex=True)
    if nrows == 1:
        axes = [axes]

    # 顶部：音频激活 + 节拍竖线
    if nrows == 2:
        ax_top = axes[0]
        t_act = np.arange(len(beat_activations), dtype=float) / float(beat_activation_fps)
        ax_top.plot(t_act, beat_activations, color="#4c72b0", lw=1.2, label="Beat Activation")
        for bt in audio_beats_s:
            ax_top.axvline(bt, color="#dd8452", alpha=0.6, ls="--", lw=1.0)
        ax_top.set_ylabel("Activation")
        ax_top.set_title("Audio Beat Activations & Beat Times")
        ax_top.legend(loc="upper right")

    # 底部：动作能量 + 峰/谷 + 节拍竖线
    ax = axes[-1]
    ax.plot(t_motion, motion_energy, color="#55a868", lw=1.2, label="Motion Energy")
    if motion_peak_idx.size > 0:
        ax.plot(t_motion[motion_peak_idx], motion_energy[motion_peak_idx], "^", color="#c44e52", label="Peaks")
    if motion_valley_idx.size > 0:
        ax.plot(t_motion[motion_valley_idx], motion_energy[motion_valley_idx], "v", color="#4c72b0", label="Valleys")
    for bt in audio_beats_s:
        ax.axvline(bt, color="#dd8452", alpha=0.4, ls=":", lw=0.9)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Energy")
    ax.set_title("Motion Energy with Peaks/Valleys & Audio Beats")
    ax.legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def build_argparser():
    parser = argparse.ArgumentParser(
        description="评估音频节拍与动作节奏的一致性 (madmom + 动作能量峰值)"
    )
    parser.add_argument("--audio", required=True, help="音频文件路径 (.wav 等)")
    parser.add_argument("--motion", required=True, help="动作特征文件路径 (.npy / .csv)")
    parser.add_argument("--fps", type=float, default=30.0, help="动作帧率，默认 30 fps")
    parser.add_argument("--tol", type=float, default=0.15, help="匹配容差（秒），默认 0.15s")
    parser.add_argument("--min_interval", type=float, default=0.2, help="动作峰的最小间隔（秒），默认 0.2s")
    parser.add_argument(
        "--prominence_scale",
        type=float,
        default=0.5,
        help="动作峰显著性的标准差比例，默认 0.5 * std",
    )
    parser.add_argument("--dbn_fps", type=int, default=100, help="madmom DBN fps，默认 100")
    parser.add_argument(
        "--save_debug_dir",
        type=str,
        default=None,
        help="可选：保存中间结果（beats/peaks/metrics）的目录",
    )
    parser.add_argument(
        "--plot_path",
        type=str,
        default=None,
        help="可选：将节奏可视化图保存到指定路径（若未给定且有 save_debug_dir，将保存到其中）",
    )
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()

    result = run_metric(
        audio_path=args.audio,
        motion_path=args.motion,
        fps_motion=args.fps,
        tolerance_s=args.tol,
        min_interval_s=args.min_interval,
        prominence_scale=args.prominence_scale,
        dbn_fps=args.dbn_fps,
        save_debug_dir=args.save_debug_dir,
        plot_path=args.plot_path,
    )

    metrics = result["metrics"]
    print("=== Beat-Motion Alignment Metrics ===")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


