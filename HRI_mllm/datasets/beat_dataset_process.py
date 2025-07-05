# 2025.07.05 HIT-xiaowangzi
# 把beat数据集里面的文本塞到text文件夹里面

import os
import subprocess
from HRI_retarget import DATA_ROOT, ROOT
import time 
from datetime import timedelta
from tqdm import tqdm

# 文件夹路径
folder_path = os.path.join(DATA_ROOT,"beat_english_v0.2.1")
save_path = os.path.join(DATA_ROOT,"G1_beat/texts")

print("folder path:", folder_path)
print("root: ", ROOT)


# 遍历文件夹及其子文件夹
todo_files = []

for root, dirs, files in os.walk(folder_path):
    # print("root:", root)
    # print("dirs:", dirs)
    num_files = len(files)
    # print("num_files:", num_files)
    
    for idx, filename in tqdm(enumerate(files)):
        # 拼写出保存路径
        if "txt" in filename:

            file_path = os.path.join(root, filename)
            temp_path = os.path.join(save_path, filename)
            print("file_path: \n", file_path)
            print("save_path: \n", temp_path)
            cmd = f"cp {file_path} {temp_path}"
            print(cmd)
            print("################\n")
            process = subprocess.Popen([cmd], shell=True)
            process.wait()

