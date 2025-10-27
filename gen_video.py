import os
os.environ['MUJOCO_GL'] = 'egl'

import subprocess
from pathlib import Path
import numpy as np

# 兼容两种可能的可视化脚本导入路径
try:
    from scripts.vis_csv_motion import vis_audio_motion
except Exception:
    from HRI_mllm.external.GMR.scripts.vis_csv_motion import vis_audio_motion


def ensure_file(path: str):
    if not Path(path).exists():
        raise FileNotFoundError(f"文件不存在: {path}")


def concat_side_by_side(video_left: str, video_right: str, output_path: str, target_height: int = 720):
    """使用 ffmpeg 横向拼接两个视频，统一高度为 target_height，保持宽高比。"""
    ensure_file(video_left)
    ensure_file(video_right)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_left,
        "-i", video_right,
        "-filter_complex",
        f"[0:v]scale=-2:{target_height},setsar=1[left];[1:v]scale=-2:{target_height},setsar=1[right];[left][right]hstack=inputs=2[v]",
        "-map", "[v]",
        "-map", "0:a?",  # 使用左视频音频（若存在）
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "veryfast",
        "-c:a", "aac",
        "-shortest",
        output_path,
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    audio_path = "audio.wav"
    ensure_file(audio_path)

    # 准备 sg.csv（来自原始 npz）
    sg_npz_path = "/root/workspace/HRI_MLLM/test_audio_1_original_motion.npz"
    ensure_file(sg_npz_path)
    sg_csv = np.load(sg_npz_path)["qpos"]
    np.savetxt("sg.csv", sg_csv, delimiter=",", fmt="%.8f")

    # 确保 llm.csv 已生成
    ensure_file("llm.csv")

    # 分别渲染两个视频
    out_llm = "final_output_llm.mp4"
    out_sg = "final_output_sg.mp4"

    vis_audio_motion("llm.csv", output_path=out_llm, audio_path=audio_path, robot_type="g1_brainco", rate_limit=False)
    vis_audio_motion("sg.csv", output_path=out_sg, audio_path=audio_path, robot_type="g1_brainco", rate_limit=False, motion_fps=60)

    # 横向拼接
    out_compare = "final_output_compare.mp4"
    concat_side_by_side(out_llm, out_sg, out_compare, target_height=720)
    print(f"已生成对比视频: {out_compare}")