"""
直接从目录加载所有motion文件的数据集类
不依赖train.txt文件，自动扫描目录中的所有.npy文件
"""
import os
import rich
import random
import pickle
import codecs as cs
import numpy as np
from torch.utils import data
from rich.progress import track
from os.path import join as pjoin
from .T2M_dataset import Text2MotionDataset


class DirectMotionDataset(Text2MotionDataset):
    """
    直接从目录加载所有motion文件的数据集类
    不依赖split文件（如train.txt），自动扫描目录中的所有.npy文件
    """
    
    def __init__(
        self,
        data_root,
        split,  # 保留split参数以保持接口一致性，但实际上不使用
        mean,
        std,
        max_motion_length=196,
        min_motion_length=40,
        unit_length=4,
        fps=20,
        tmpFile=True,
        tiny=False,
        debug=False,
        **kwargs,
    ):
        dataset_name = kwargs.get("dataset_name", "DirectMotion")
        
        # 限制motion和text的长度
        self.max_length = 20
        self.max_motion_length = max_motion_length
        self.min_motion_length = min_motion_length
        self.unit_length = unit_length
        
        # 数据均值和标准差
        self.mean = mean
        self.std = std
        
        # 数据路径
        motion_dir = pjoin(data_root, 'new_joint_vecs')
        
        # 🔧 直接从目录扫描所有.npy文件，不依赖split文件
        if not os.path.exists(motion_dir):
            raise FileNotFoundError(f"❌ Motion directory not found: {motion_dir}")
        
        # 获取所有.npy文件
        all_files = os.listdir(motion_dir)
        npy_files = [f for f in all_files if f.endswith('.npy')]
        
        # 提取文件名（不含扩展名）作为id_list
        self.id_list = [os.path.splitext(f)[0] for f in npy_files]
        
        if len(self.id_list) == 0:
            raise ValueError(f"❌ No .npy files found in {motion_dir}")
        
        print(f"✅ Found {len(self.id_list)} motion files in {motion_dir}")
        
        # Debug模式
        if tiny or debug:
            enumerator = enumerate(self.id_list)
            maxdata = 100
            subset = '_tiny'
        else:
            enumerator = enumerate(
                track(
                    self.id_list,
                    f"Loading DirectMotion {split}",
                ))
            maxdata = 1e10
            subset = ''
        
        new_name_list = []
        length_list = []
        data_dict = {}
        
        # Fast loading
        # 🔧 使用direct_motion作为缓存文件名，避免与其他数据集冲突
        cache_file = pjoin(data_root, f'tmp/direct_motion{split}{subset}_data.pkl')
        if os.path.exists(cache_file) and tmpFile:
            if tiny or debug:
                with open(cache_file, 'rb') as file:
                    data_dict = pickle.load(file)
            else:
                with rich.progress.open(
                        cache_file,
                        'rb',
                        description=f"Loading DirectMotion {split}") as file:
                    data_dict = pickle.load(file)
            index_file = pjoin(data_root, f'tmp/direct_motion{split}{subset}_index.pkl')
            with open(index_file, 'rb') as file:
                name_list = pickle.load(file)
            for name in name_list:
                length_list.append(data_dict[name]['length'])
        else:
            # 加载所有motion文件
            for idx, name in enumerator:
                if len(new_name_list) > maxdata:
                    break
                try:
                    motion_file = pjoin(motion_dir, name + ".npy")
                    if not os.path.exists(motion_file):
                        print(f"⚠️  File not found: {motion_file}, skipping")
                        continue
                    
                    motion = np.load(motion_file)
                    
                    # 🔧 对于DirectMotionDataset，保留所有动作（包括过短的）
                    # 只检查最小合理长度（确保至少有几帧，比如8帧）
                    min_required_length = 8  # 最小必需长度，确保可以处理
                    if len(motion) < min_required_length:
                        print(f"⚠️  Motion {name} is too short ({len(motion)} frames), skipping")
                        continue
                    
                    # 🔧 对于DirectMotionDataset，允许更长的序列（最多512帧）
                    # 如果超过max_motion_length但不超过512，仍然保留
                    max_allowed_length = max(self.max_motion_length, 512)
                    if len(motion) > max_allowed_length:
                        # 如果超过最大允许长度，跳过
                        print(f"⚠️  Motion {name} is too long ({len(motion)} frames), skipping")
                        continue
                    
                    # 直接使用文件名作为key
                    data_dict[name] = {
                        'motion': motion,
                        "length": len(motion),
                        'text': []  # 没有text数据，使用空列表
                    }
                    new_name_list.append(name)
                    length_list.append(len(motion))
                    
                except Exception as e:
                    print(f"⚠️  Error processing {name}: {e}")
                    continue
            
            # 按长度排序
            if new_name_list:
                name_list, length_list = zip(
                    *sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
            else:
                name_list = []
                length_list = []
            
            # 保存缓存
            if tmpFile and name_list:
                os.makedirs(pjoin(data_root, 'tmp'), exist_ok=True)
                with open(cache_file, 'wb') as file:
                    pickle.dump(data_dict, file)
                index_file = pjoin(data_root, f'tmp/direct_motion{split}{subset}_index.pkl')
                with open(index_file, 'wb') as file:
                    pickle.dump(name_list, file)
        
        if not name_list:
            raise ValueError(f"❌ No valid motion files found in {motion_dir}")
        
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = list(name_list)
        self.nfeats = data_dict[self.name_list[0]]['motion'].shape[1]
        self.reset_max_len(self.max_length)
        
        print(f"✅ Loaded {len(self.name_list)} valid motion files")

