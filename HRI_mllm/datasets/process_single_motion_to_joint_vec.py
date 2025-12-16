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

# 添加HRI_retarget到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'HRI_mllm/external/HRI_retarget'))

# 使用优化后的快速FK计算模块（不使用梯度，速度更快）
from HRI_mllm.datasets.fast_fk_brainco import fast_data_pkl_to_vec as data_pkl_to_vec, fast_data_pkl_to_vec_batch

# 解析命令行参数
parser = argparse.ArgumentParser(description='处理motion数据转换为joint_vec')
parser.add_argument('--input_dir', type=str, 
                    default="/root/workspace/HRI_MLLM/data/synthetic_data/SG_2_or_3_long_sentence_1023_cn",
                    help='输入数据目录路径（递归扫描子目录，适配synthetic_data）')
parser.add_argument('--output_dir', type=str, default=None,
                    help='输出目录路径（默认：输入目录名_joint_vecs）')
parser.add_argument('--batch_size', type=int, default=8,
                    help='批量处理大小，可以根据GPU内存调整（默认：8）')
parser.add_argument('--num_gpus', type=int, default=8,
                    help='使用的GPU数量上限（默认：8，会自动截断为实际可用数）')
parser.add_argument('--num_threads', type=int, default=16,
                    help='线程数量（默认：16，建议 >= GPU 数 * 2）')
args = parser.parse_args()

# 数据集路径（将在运行时按目录动态更新）
folder_path = args.input_dir
if args.output_dir is None:
    tgt_dir = folder_path + "_joint_vecs"
else:
    tgt_dir = args.output_dir
# 创建3个子目录分别存放npy、json、wav文件（在每次处理目录时更新）
npy_dir = os.path.join(tgt_dir, "npy")
json_dir = os.path.join(tgt_dir, "json")
wav_dir = os.path.join(tgt_dir, "wav")

# 全局变量：开始时间（将在主程序中初始化）
starting_time = None

# 多线程配置
try:
    import torch
    _avail_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
except Exception:
    _avail_gpu = 0

NUM_GPUS = max(1, min(args.num_gpus, _avail_gpu) if _avail_gpu > 0 else 1)
NUM_THREADS = max(1, args.num_threads)  # 用户可自定义线程数
task_queue = Queue()
success_lock = Lock()
fail_lock = Lock()
success_count = 0
fail_count = 0


def format_eta(elapsed_seconds, finished, total):
    """根据已完成数量估算剩余时间"""
    if finished <= 0 or total <= 0:
        return "未知"
    rate = finished / max(elapsed_seconds, 1e-6)
    remaining = max(total - finished, 0)
    eta_seconds = int(remaining / rate) if rate > 0 else -1
    return str(timedelta(seconds=eta_seconds)).split(".")[0] if eta_seconds >= 0 else "未知"


def find_npz_files(root_dir):
    """查找所有.npz文件"""
    root_path = Path(root_dir)
    npz_files = list(root_path.rglob('*.npz'))
    return [str(file) for file in npz_files]


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
        # 例如: HAND_CALL/blend_npz_0.4_60_60_linear_with_annotation/1.npz
        # 转换为: HAND_CALL_blend_npz_0.4_60_60_linear_with_annotation_1.npy
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
        wav_source = os.path.join(source_dir, f"{source_base}.wav")
        if os.path.exists(wav_source):
            save_name_wav = base_name.replace(".npz", ".wav")
            save_path_wav = os.path.join(wav_dir, save_name_wav)
            shutil.copy2(wav_source, save_path_wav)
        
        return True
    except Exception as e:
        print(f"处理文件 {source_file} 时出错: {e}")
        return False


def process_batch_npz_to_joint_vec(file_batch, tgt_dir, folder_path, gpu_id):
    """
    批量处理多个npz文件，转换为joint_vec
    
    Args:
        file_batch: list of (idx, source_file) tuples
        tgt_dir: 输出目录
        folder_path: 输入目录
        gpu_id: GPU ID
    
    Returns:
        list of (idx, success) tuples
    """
    results = []
    
    try:
        # 加载所有文件的数据
        data_dict_list = []
        file_info_list = []  # 保存文件信息用于后续保存
        
        for idx, source_file in file_batch:
            try:
                data = np.load(source_file)
                qpos = data['qpos']
                dof = qpos[:, 7:]
                
                data_dict = {
                    "angles": dof,
                    "robot_name": "g1_brainco",
                    "fps": 60
                }
                
                data_dict_list.append(data_dict)
                file_info_list.append((idx, source_file))
            except Exception as e:
                print(f"加载文件 {source_file} 时出错: {e}")
                results.append((idx, False))
        
        if len(data_dict_list) == 0:
            return results
        
        # 批量转换为joint_vec
        vec_list = fast_data_pkl_to_vec_batch(data_dict_list)
        
        # 保存结果
        for (idx, source_file), vec in zip(file_info_list, vec_list):
            try:
                # 获取相对路径并重命名
                relative_path = os.path.relpath(source_file, folder_path)
                base_name = relative_path.replace(os.sep, "_")
                save_name_npy = base_name.replace(".npz", ".npy")
                
                # 保存joint_vec文件
                save_path_npy = os.path.join(npy_dir, save_name_npy)
                np.save(save_path_npy, vec)
                
                # 拷贝并重命名对应的json和wav文件
                source_dir = os.path.dirname(source_file)
                source_base = os.path.basename(source_file).replace(".npz", "")
                
                # 处理json文件
                json_source = os.path.join(source_dir, f"{source_base}.json")
                if os.path.exists(json_source):
                    save_name_json = base_name.replace(".npz", ".json")
                    save_path_json = os.path.join(json_dir, save_name_json)
                    shutil.copy2(json_source, save_path_json)
                
                # 处理wav文件
                wav_source = os.path.join(source_dir, f"{source_base}.wav")
                if os.path.exists(wav_source):
                    save_name_wav = base_name.replace(".npz", ".wav")
                    save_path_wav = os.path.join(wav_dir, save_name_wav)
                    shutil.copy2(wav_source, save_path_wav)
                
                results.append((idx, True))
            except Exception as e:
                print(f"保存文件 {source_file} 时出错: {e}")
                results.append((idx, False))
    
    except Exception as e:
        print(f"批量处理出错: {e}")
        # 如果批量处理失败，标记所有为失败
        for idx, _ in file_batch:
            if (idx, False) not in results and (idx, True) not in results:
                results.append((idx, False))
    
    return results


def worker(worker_id, total_num, batch_size):
    """工作线程函数，支持批量处理"""
    global success_count, fail_count
    gpu_id = worker_id % NUM_GPUS
    
    # 在每个worker线程中设置CUDA设备（需要在导入torch相关模块之前）
    # 注意：由于环境变量是进程级别的，这里使用torch.cuda.set_device
    import torch
    if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
        torch.cuda.set_device(gpu_id)
    
    # 收集批次
    batch = []
    
    while True:
        try:
            # 从队列获取任务，超时1秒
            item = task_queue.get(timeout=1)
            if item is None:  # 结束信号
                # 处理剩余的批次
                if len(batch) > 0:
                    batch_results = process_batch_npz_to_joint_vec(batch, tgt_dir, folder_path, gpu_id)
                    for idx, success in batch_results:
                        if success:
                            with success_lock:
                                success_count += 1
                        else:
                            with fail_lock:
                                fail_count += 1
                break
            
            idx, source_file = item
            batch.append((idx, source_file))
            
            # 当批次达到指定大小时，批量处理
            if len(batch) >= batch_size:
                batch_results = process_batch_npz_to_joint_vec(batch, tgt_dir, folder_path, gpu_id)
                for result_idx, success in batch_results:
                    if success:
                        with success_lock:
                            success_count += 1
                    else:
                        with fail_lock:
                            fail_count += 1
                    
                    # 每处理100个文件输出一次进度
                    if result_idx % 100 == 0:
                        elapsed = time.time() - starting_time
                        finished = success_count + fail_count
                        eta = format_eta(elapsed, finished, total_num)
                        print(f"进度: {finished}/{total_num}, 成功: {success_count}, 失败: {fail_count}, "
                              f"耗时: {str(timedelta(seconds=int(elapsed))).split('.')[0]}, 预计剩余: {eta}")
                
                batch = []  # 清空批次
            
            task_queue.task_done()
        except Exception as e:
            if "Empty" not in str(e):  # 忽略队列为空的异常
                print(f"Worker {worker_id} 出错: {e}")
                # 处理当前批次
                if len(batch) > 0:
                    try:
                        batch_results = process_batch_npz_to_joint_vec(batch, tgt_dir, folder_path, gpu_id)
                        for idx, success in batch_results:
                            if success:
                                with success_lock:
                                    success_count += 1
                            else:
                                with fail_lock:
                                    fail_count += 1
                    except:
                        pass
            break


if __name__ == "__main__":
    def process_directory(source_dir, custom_output_dir=None):
        """
        针对指定目录单独生成 joint_vec 结果；会重置全局状态，避免不同子目录混写。
        """
        global folder_path, tgt_dir, npy_dir, json_dir, wav_dir
        global task_queue, success_lock, fail_lock, success_count, fail_count, starting_time
        
        folder_path = source_dir
        if custom_output_dir is None:
            tgt_dir = folder_path + "_joint_vecs"
        else:
            tgt_dir = custom_output_dir
        npy_dir = os.path.join(tgt_dir, "npy")
        json_dir = os.path.join(tgt_dir, "json")
        wav_dir = os.path.join(tgt_dir, "wav")
        
        task_queue = Queue()
        success_lock = Lock()
        fail_lock = Lock()
        success_count = 0
        fail_count = 0
        starting_time = time.time()
        
        # 创建目标目录和子目录
        os.makedirs(tgt_dir, exist_ok=True)
        os.makedirs(npy_dir, exist_ok=True)
        os.makedirs(json_dir, exist_ok=True)
        os.makedirs(wav_dir, exist_ok=True)
        
        # 查找所有npz文件
        print(f"\n正在处理目录: {folder_path}")
        print("正在查找所有.npz文件...")
        npz_files = find_npz_files(folder_path)
        total_num = len(npz_files)
        print(f"找到 {total_num} 个.npz文件")
        
        if total_num == 0:
            print("未找到npz文件，跳过该目录。")
            return
        
        # 将所有文件加入任务队列
        batch_size = args.batch_size
        print(f"使用 {NUM_GPUS} 个GPU, {NUM_THREADS} 个线程进行处理...")
        print(f"批量处理大小: {batch_size}")
        for idx, source_file in enumerate(npz_files, 1):
            task_queue.put((idx, source_file))
        
        # 启动工作线程
        threads = []
        for i in range(NUM_THREADS):
            t = Thread(target=worker, args=(i, total_num, batch_size))
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
    
    # 判断输入目录是否直接包含npz；否则对一级子目录逐个处理
    input_path = Path(args.input_dir)
    has_npz = len(list(input_path.glob("*.npz"))) > 0
    
    if has_npz:
        # 直接处理当前目录
        process_directory(str(input_path), args.output_dir)
    else:
        subdirs = [p for p in input_path.iterdir() if p.is_dir()]
        if len(subdirs) == 0:
            print("输入目录下没有npz文件或子目录，退出。")
        else:
            processed_any = False
            for sub in sorted(subdirs):
                # 判断子目录是否含npz
                if len(list(sub.rglob("*.npz"))) == 0:
                    continue
                processed_any = True
                if args.output_dir is None:
                    out_dir = None  # 默认：子目录后缀 _joint_vecs
                else:
                    # 在指定输出根目录下为每个子目录创建独立输出
                    out_dir = os.path.join(args.output_dir, sub.name + "_joint_vecs")
                process_directory(str(sub), out_dir)
            
            if not processed_any:
                print("未在任何子目录中找到npz文件，退出。")

