import os
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time
from datetime import timedelta
from queue import Queue
from threading import Thread, Lock
import threading
import shutil
import argparse
import re

# 添加HRI_retarget到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../external/HRI_retarget'))

from HRI_retarget.utils.io.brainco_representation import data_pkl_to_vec

# 解析命令行参数
parser = argparse.ArgumentParser(description='处理motion数据转换为joint_vec（指定编号范围）')
parser.add_argument('--input_dir', type=str, 
                    default="/root/workspace/HRI_MLLM/data/HAND_RING_1111",
                    help='输入数据目录路径')
parser.add_argument('--output_dir', type=str, default=None,
                    help='输出目录路径（默认：输入目录名_joint_vecs_0_1000）')
parser.add_argument('--start_idx', type=int, default=0,
                    help='起始编号（默认：0）')
parser.add_argument('--end_idx', type=int, default=1000,
                    help='结束编号（默认：1000）')
args = parser.parse_args()

# 数据集路径
folder_path = args.input_dir
if args.output_dir is None:
    # 如果没有指定输出目录，则使用输入目录名加上范围后缀
    tgt_dir = folder_path + f"_joint_vecs_{args.start_idx}_{args.end_idx}"
else:
    tgt_dir = args.output_dir
    
# 创建3个子目录分别存放npy、json、wav文件
npy_dir = os.path.join(tgt_dir, "npy")
json_dir = os.path.join(tgt_dir, "json")
wav_dir = os.path.join(tgt_dir, "wav")

# 全局变量：开始时间（将在主程序中初始化）
starting_time = None

# 多线程配置
NUM_GPUS = 8
NUM_THREADS = 16  # 每个GPU 2个线程
task_queue = Queue()
success_lock = Lock()
fail_lock = Lock()
success_count = 0
fail_count = 0


def extract_file_number(filepath):
    """从文件路径中提取数字编号
    例如: /path/to/1000_B.npz -> 1000
         /path/to/500_A.npz -> 500
    """
    basename = os.path.basename(filepath)
    # 匹配 数字_字母.npz 格式
    match = re.search(r'^(\d+)_[A-Za-z]\.npz$', basename)
    if match:
        return int(match.group(1))
    return None


def find_npz_files_in_range(root_dir, start_idx, end_idx):
    """查找指定编号范围内的.npz文件"""
    root_path = Path(root_dir)
    all_npz_files = list(root_path.rglob('*.npz'))
    
    # 过滤出指定范围内的文件
    filtered_files = []
    skipped_files = []
    
    for file in all_npz_files:
        file_num = extract_file_number(str(file))
        if file_num is not None:
            if start_idx <= file_num <= end_idx:
                filtered_files.append(str(file))
            else:
                skipped_files.append(str(file))
        else:
            # 无法提取编号的文件
            skipped_files.append(str(file))
    
    print(f"找到 {len(all_npz_files)} 个.npz文件")
    print(f"编号在 [{start_idx}, {end_idx}] 范围内的文件: {len(filtered_files)} 个")
    print(f"跳过的文件: {len(skipped_files)} 个")
    
    return filtered_files


def process_npz_to_joint_vec(source_file, tgt_dir, folder_path, gpu_id):
    """处理单个npz文件，转换为joint_vec，并拷贝对应的json和wav文件"""
    try:
        # 加载npz文件
        data = np.load(source_file)
        qpos = data['qpos']
        
        # 提取dof（从第7列开始，共53个自由度）
        dof = qpos[:, 7:]
        
        # 构建data_dict
        data_dict = {
            "angles": dof,
            "robot_name": "g1_brainco",
            "fps": 60
        }
        
        # 转换为joint_vec
        vec = data_pkl_to_vec(data_dict)
        
        # 获取相对路径，并根据路径重命名文件
        relative_path = os.path.relpath(source_file, folder_path)
        # 将路径分隔符替换为下划线，并去掉.npz扩展名
        base_name = relative_path.replace(os.sep, "_")
        save_name_npy = base_name.replace(".npz", ".npy")
        
        # 保存joint_vec文件到npy目录
        save_path_npy = os.path.join(npy_dir, save_name_npy)
        np.save(save_path_npy, vec)
        
        # 拷贝并重命名对应的json和wav文件
        source_dir = os.path.dirname(source_file)
        source_base = os.path.basename(source_file).replace(".npz", "")
        
        # 处理json文件，保存到json目录
        json_source = os.path.join(source_dir, f"{source_base}.json")
        if os.path.exists(json_source):
            save_name_json = base_name.replace(".npz", ".json")
            save_path_json = os.path.join(json_dir, save_name_json)
            shutil.copy2(json_source, save_path_json)
        
        # 处理wav文件，保存到wav目录
        # 获取文件编号（例如从 1000_B.npz 中提取 1000）
        file_num = extract_file_number(source_file)
        if file_num is not None:
            # 查找所有同编号的wav文件（如 1000_A.wav, 1000_B.wav 等）
            import glob
            wav_pattern = os.path.join(source_dir, f"{file_num}_*.wav")
            wav_files = glob.glob(wav_pattern)
            
            for wav_file in wav_files:
                # 获取wav文件的基础名（如 1000_A.wav）
                wav_basename = os.path.basename(wav_file)
                # 构造保存路径，保持原有的目录结构前缀
                wav_relative_path = os.path.relpath(wav_file, folder_path)
                save_name_wav = wav_relative_path.replace(os.sep, "_")
                save_path_wav = os.path.join(wav_dir, save_name_wav)
                shutil.copy2(wav_file, save_path_wav)
        else:
            # 如果无法提取编号，使用原来的逻辑（向后兼容）
            wav_source = os.path.join(source_dir, f"{source_base}.wav")
            if os.path.exists(wav_source):
                save_name_wav = base_name.replace(".npz", ".wav")
                save_path_wav = os.path.join(wav_dir, save_name_wav)
                shutil.copy2(wav_source, save_path_wav)
        
        return True
    except Exception as e:
        print(f"处理文件 {source_file} 时出错: {e}")
        return False


def worker(worker_id, total_num):
    """工作线程函数"""
    global success_count, fail_count
    gpu_id = worker_id % NUM_GPUS
    
    # 在每个worker线程中设置CUDA设备（需要在导入torch相关模块之前）
    # 注意：由于环境变量是进程级别的，这里使用torch.cuda.set_device
    import torch
    if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
        torch.cuda.set_device(gpu_id)
    
    while True:
        try:
            # 从队列获取任务，超时1秒
            item = task_queue.get(timeout=1)
            if item is None:  # 结束信号
                break
            
            idx, source_file = item
            
            # 处理文件
            if process_npz_to_joint_vec(source_file, tgt_dir, folder_path, gpu_id):
                with success_lock:
                    success_count += 1
            else:
                with fail_lock:
                    fail_count += 1
            
            # 每处理100个文件输出一次进度
            if idx % 100 == 0:
                elapsed = time.time() - starting_time
                print(f"进度: {idx}/{total_num}, 成功: {success_count}, 失败: {fail_count}, "
                      f"耗时: {str(timedelta(seconds=int(elapsed))).split('.')[0]}")
            
            task_queue.task_done()
        except Exception as e:
            if "Empty" not in str(e):  # 忽略队列为空的异常
                print(f"Worker {worker_id} 出错: {e}")
            break


if __name__ == "__main__":
    starting_time = time.time()
    
    # 打印配置信息
    print(f"输入目录: {folder_path}")
    print(f"输出目录: {tgt_dir}")
    print(f"处理编号范围: [{args.start_idx}, {args.end_idx}]")
    
    # 创建目标目录和子目录
    os.makedirs(tgt_dir, exist_ok=True)
    os.makedirs(npy_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(wav_dir, exist_ok=True)
    
    # 查找指定范围内的npz文件
    print("正在查找指定范围内的.npz文件...")
    npz_files = find_npz_files_in_range(folder_path, args.start_idx, args.end_idx)
    total_num = len(npz_files)
    
    if total_num == 0:
        print("未找到符合条件的文件！")
        sys.exit(0)
    
    print(f"将处理 {total_num} 个文件")
    
    # 将所有文件加入任务队列
    print(f"使用 {NUM_GPUS} 个GPU, {NUM_THREADS} 个线程进行处理...")
    for idx, source_file in enumerate(npz_files, 1):
        task_queue.put((idx, source_file))
    
    # 启动工作线程
    threads = []
    for i in range(NUM_THREADS):
        t = Thread(target=worker, args=(i, total_num))
        t.start()
        threads.append(t)
    
    # 等待所有任务完成
    task_queue.join()
    
    # 发送结束信号给所有线程
    for _ in range(NUM_THREADS):
        task_queue.put(None)
    
    # 等待所有线程结束
    for t in threads:
        t.join()
    
    # 输出统计信息
    elapsed_time = time.time() - starting_time
    print(f"\n处理完成！")
    print(f"成功: {success_count} 个文件")
    print(f"失败: {fail_count} 个文件")
    print(f"总耗时: {timedelta(seconds=int(elapsed_time))}")
    print(f"输出目录: {tgt_dir}")

