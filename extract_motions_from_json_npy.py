#!/usr/bin/env python3
"""
根据 JSON 文件从 NPY 文件中提取对应的 motion 数据
每个 motion 提取前后4秒，共计100帧（25 FPS）
"""

import numpy as np
import json
import os
import argparse
from pathlib import Path
try:
    from tqdm import tqdm
except ImportError:
    # 如果没有 tqdm，使用简单的进度显示
    def tqdm(iterable, **kwargs):
        return iterable


def extract_motions_from_json_npy(npy_path, json_path, output_dir=None, 
                                   extract_duration=4.0, npy_fps=25, verbose=False):
    """
    从 NPY 文件中提取 JSON 中定义的 motion 数据
    
    Args:
        npy_path: NPY 文件路径
        json_path: JSON 文件路径
        output_dir: 输出目录（默认为 NPY 文件所在目录的 extracted_motions 子目录）
        extract_duration: 提取时长（秒），默认 4.0 秒
        npy_fps: NPY 文件的 FPS，默认 25
        verbose: 是否显示详细输出，默认 False
    
    Returns:
        extracted_motions: 提取的 motion 信息列表
    """
    # 读取数据
    if verbose:
        print(f"正在加载数据...")
        print(f"  NPY 文件: {npy_path}")
        print(f"  JSON 文件: {json_path}")
    
    npy_data = np.load(npy_path)
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    if verbose:
        print(f"NPY 数据形状: {npy_data.shape}")
        print(f"JSON 总时长: {json_data['total_duration']} 秒")
        print(f"Motion 数量: {json_data['num_motions']}")
    
    extract_frames = int(extract_duration * npy_fps)
    if verbose:
        print(f"提取参数: {extract_duration} 秒 = {extract_frames} 帧 (FPS={npy_fps})")
        print()
    
    # 创建输出目录和子文件夹
    if output_dir is None:
        npy_dir = os.path.dirname(npy_path)
        output_dir = os.path.join(npy_dir, 'extracted_motions')
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建 npy 和 jsonl 子文件夹
    npy_output_dir = os.path.join(output_dir, 'npy')
    jsonl_output_dir = os.path.join(output_dir, 'jsonl')
    os.makedirs(npy_output_dir, exist_ok=True)
    os.makedirs(jsonl_output_dir, exist_ok=True)
    
    # 获取基础文件名（不含扩展名）
    base_name = Path(npy_path).stem
    
    # 提取每个 motion
    extracted_motions = []
    
    for i, motion in enumerate(json_data['blended_timeline']):
        motion_name = motion['motion']
        
        # 计算 motion 的中心时间（使用 actual 时间范围）
        center_time = (motion['actual_start_time'] + motion['actual_end_time']) / 2.0
        
        # 计算提取的时间范围（前后各 extract_duration/2 秒）
        extract_start_time = center_time - extract_duration / 2.0
        extract_end_time = center_time + extract_duration / 2.0
        
        # 转换为帧索引
        start_frame = int(extract_start_time * npy_fps)
        end_frame = start_frame + extract_frames
        
        # 边界检查
        original_start = start_frame
        original_end = end_frame
        
        if start_frame < 0:
            if verbose:
                print(f"⚠️  Motion {i+1} ({motion_name}): 开始帧 {start_frame} < 0，调整为 0")
            start_frame = 0
            end_frame = extract_frames
        
        if end_frame > len(npy_data):
            if verbose:
                print(f"⚠️  Motion {i+1} ({motion_name}): 结束帧 {end_frame} > 总帧数 {len(npy_data)}，调整为 {len(npy_data)}")
            end_frame = len(npy_data)
            start_frame = end_frame - extract_frames
            if start_frame < 0:
                start_frame = 0
        
        # 提取数据
        motion_data = npy_data[start_frame:end_frame]
        
        # 如果提取的帧数不足，进行零填充
        if len(motion_data) < extract_frames:
            if verbose:
                print(f"⚠️  Motion {i+1} ({motion_name}): 提取的帧数 {len(motion_data)} < {extract_frames}，进行零填充")
            padding = np.zeros((extract_frames - len(motion_data), motion_data.shape[1]), 
                             dtype=motion_data.dtype)
            motion_data = np.vstack([motion_data, padding])
        elif len(motion_data) > extract_frames:
            motion_data = motion_data[:extract_frames]
        
        # 保存提取的数据到 npy 子文件夹
        safe_motion_name = motion_name.replace('-', '_').replace(' ', '_')
        output_filename = f"{base_name}_motion_{i+1}_{safe_motion_name}.npy"
        output_path = os.path.join(npy_output_dir, output_filename)
        np.save(output_path, motion_data)
        
        # 记录信息
        info = {
            'source_file': base_name,
            'motion_index': i + 1,
            'motion_name': motion_name,
            'center_time': center_time,
            'extract_start_time': extract_start_time,
            'extract_end_time': extract_end_time,
            'start_frame': start_frame,
            'end_frame': end_frame,
            'extracted_frames': len(motion_data),
            'npy_path': output_path,
            'npy_filename': output_filename,
            'original_start_time': motion['actual_start_time'],
            'original_end_time': motion['actual_end_time'],
            'original_text': motion.get('text', '')
        }
        extracted_motions.append(info)
        
        if verbose:
            print(f"Motion {i+1}: {motion_name}")
            print(f"  中心时间: {center_time:.3f} 秒")
            print(f"  提取时间范围: [{extract_start_time:.3f}, {extract_end_time:.3f}] 秒")
            print(f"  NPY 索引范围: [{start_frame}, {end_frame})")
            print(f"  提取帧数: {len(motion_data)}")
            print(f"  保存路径: {output_path}")
            print()
    
    # 保存提取信息到 JSONL 文件（每行一个 motion 的 JSON）
    jsonl_path = os.path.join(jsonl_output_dir, f'{base_name}_motions.jsonl')
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for motion_info in extracted_motions:
            # 将每个 motion 信息写入一行 JSON
            json.dump(motion_info, f, ensure_ascii=False)
            f.write('\n')
    
    # 同时保存汇总信息到 JSON 文件（可选，用于调试）
    summary_path = os.path.join(jsonl_output_dir, f'{base_name}_extraction_summary.json')
    extraction_summary = {
        'source_npy': npy_path,
        'source_json': json_path,
        'npy_fps': npy_fps,
        'extract_duration': extract_duration,
        'extract_frames': extract_frames,
        'total_motions': len(extracted_motions),
        'npy_output_dir': npy_output_dir,
        'jsonl_output_dir': jsonl_output_dir
    }
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(extraction_summary, f, indent=2, ensure_ascii=False)
    
    if verbose:
        print(f"✅ 提取完成！")
        print(f"NPY 文件保存到: {npy_output_dir}")
        print(f"JSONL 文件保存到: {jsonl_path}")
        print(f"汇总信息保存到: {summary_path}")
        print(f"共提取 {len(extracted_motions)} 个 motion")
        print(f"输出目录: {output_dir}")
    
    return extracted_motions


def batch_process(npy_dir, json_dir, output_dir=None, 
                 extract_duration=4.0, npy_fps=25):
    """
    批量处理 npy 和 json 文件夹中的所有文件
    
    Args:
        npy_dir: NPY 文件目录
        json_dir: JSON 文件目录
        output_dir: 输出目录（默认为 npy_dir 的 extracted_motions 子目录）
        extract_duration: 提取时长（秒），默认 4.0 秒
        npy_fps: NPY 文件的 FPS，默认 25
    """
    npy_dir = Path(npy_dir)
    json_dir = Path(json_dir)
    
    if not npy_dir.exists():
        raise FileNotFoundError(f"NPY 目录不存在: {npy_dir}")
    if not json_dir.exists():
        raise FileNotFoundError(f"JSON 目录不存在: {json_dir}")
    
    # 获取所有 npy 文件
    npy_files = sorted(npy_dir.glob("*.npy"))
    json_files = sorted(json_dir.glob("*.json"))
    
    print(f"找到 {len(npy_files)} 个 NPY 文件")
    print(f"找到 {len(json_files)} 个 JSON 文件")
    print()
    
    # 创建输出目录
    if output_dir is None:
        output_dir = npy_dir.parent / 'extracted_motions'
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 统计信息
    total_processed = 0
    total_failed = 0
    failed_files = []
    total_motions = 0
    
    # 处理每个 npy 文件（使用进度条）
    pbar = tqdm(npy_files, desc="处理文件", unit="file")
    for npy_path in pbar:
        npy_stem = npy_path.stem
        json_path = json_dir / f"{npy_stem}.json"
        
        # 更新进度条描述
        pbar.set_postfix(file=npy_path.name)
        
        if not json_path.exists():
            total_failed += 1
            failed_files.append(npy_path.name)
            continue
        
        try:
            extracted_motions = extract_motions_from_json_npy(
                npy_path=str(npy_path),
                json_path=str(json_path),
                output_dir=str(output_dir),
                extract_duration=extract_duration,
                npy_fps=npy_fps,
                verbose=False  # 批量处理时不显示详细输出
            )
            
            total_processed += 1
            total_motions += len(extracted_motions)
            
        except Exception as e:
            total_failed += 1
            failed_files.append(npy_path.name)
            tqdm.write(f"❌ 处理 {npy_path.name} 时出错: {e}")
    
    # 打印总结
    print()
    print("=" * 80)
    print("批量处理完成！")
    print("=" * 80)
    print(f"成功处理: {total_processed} 个文件")
    print(f"共提取: {total_motions} 个 motion")
    print(f"失败: {total_failed} 个文件")
    if failed_files:
        print(f"失败的文件: {', '.join(failed_files[:10])}")  # 只显示前10个
        if len(failed_files) > 10:
            print(f"  ... 还有 {len(failed_files) - 10} 个失败文件")
    print(f"输出目录: {output_dir}")
    print(f"  - NPY 文件: {output_dir / 'npy'}")
    print(f"  - JSONL 文件: {output_dir / 'jsonl'}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='根据 JSON 文件从 NPY 文件中提取对应的 motion 数据'
    )
    parser.add_argument('--npy', type=str, default=None,
                       help='NPY 文件路径或目录路径')
    parser.add_argument('--json', type=str, default=None,
                       help='JSON 文件路径或目录路径')
    parser.add_argument('--npy_dir', type=str, default=None,
                       help='NPY 文件目录（批量处理模式）')
    parser.add_argument('--json_dir', type=str, default=None,
                       help='JSON 文件目录（批量处理模式）')
    parser.add_argument('--output', type=str, default=None,
                       help='输出目录（默认为 NPY 文件所在目录的 extracted_motions 子目录）')
    parser.add_argument('--duration', type=float, default=4.0,
                       help='提取时长（秒），默认 4.0 秒')
    parser.add_argument('--fps', type=int, default=25,
                       help='NPY 文件的 FPS，默认 25')
    parser.add_argument('--verbose', action='store_true',
                       help='显示详细输出（单文件模式默认开启）')
    
    args = parser.parse_args()
    
    # 判断是批量处理还是单文件处理
    use_batch = False
    
    # 如果指定了 npy_dir 和 json_dir，使用批量处理
    if args.npy_dir and args.json_dir:
        use_batch = True
    # 如果 --npy 和 --json 都是目录，也使用批量处理
    elif args.npy and args.json:
        npy_path = Path(args.npy)
        json_path = Path(args.json)
        if npy_path.is_dir() and json_path.is_dir():
            use_batch = True
            args.npy_dir = str(npy_path)
            args.json_dir = str(json_path)
    
    if use_batch:
        # 批量处理模式
        npy_dir = args.npy_dir or args.npy
        json_dir = args.json_dir or args.json
        
        if not npy_dir or not json_dir:
            parser.error("批量处理模式需要指定 --npy_dir 和 --json_dir，或者 --npy 和 --json 都指向目录")
        
        batch_process(
            npy_dir=npy_dir,
            json_dir=json_dir,
            output_dir=args.output,
            extract_duration=args.duration,
            npy_fps=args.fps
        )
    else:
        # 单文件处理模式
        if not args.npy or not args.json:
            parser.error("单文件处理模式需要指定 --npy 和 --json（文件路径）")
        
        # 单文件模式默认显示详细输出
        extract_motions_from_json_npy(
            npy_path=args.npy,
            json_path=args.json,
            output_dir=args.output,
            extract_duration=args.duration,
            npy_fps=args.fps,
            verbose=True  # 单文件模式默认显示详细输出
        )


if __name__ == '__main__':
    main()

