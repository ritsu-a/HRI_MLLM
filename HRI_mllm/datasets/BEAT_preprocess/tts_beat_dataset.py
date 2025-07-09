import os
import subprocess
from HRI_mllm import DATA_ROOT, ROOT
from tqdm import tqdm
import time 
from datetime import timedelta
from queue import Queue
from threading import Thread
import numpy as np

# 文件夹路径
folder_path = os.path.join(DATA_ROOT,"beat_english_v0.2.1")
starting_time = time.time()


# 遍历文件夹及其子文件夹
todo_files = []
task_queue = Queue()

for root, dirs, files in os.walk(folder_path):
    num_files = len(files)
    
    for idx, filename in tqdm(enumerate(files)):
        # 获取文件的完整路径
        file_path = os.path.join(root, filename)


        # 使用subprocess执行Shell命令
        # 示例：使用wc命令统计文件行数
        if "TextGrid" in file_path:
            # result = subprocess.run(["python", os.path.join(ROOT,"retarget/beat_g1_inspirehands.py"), file_path], capture_output=True, text=True)
            # # 打印命令输出
            # print("Command output:")
            # print(result.stdout)

            # # 如果命令执行失败，打印错误信息
            # if result.returncode != 0:
            #     print("Error:", result.stderr)    
            file_path = file_path.replace(".TextGrid", ".wav")
            todo_files.append(file_path) 
            task_queue.put((len(todo_files),file_path))

total_num = len(todo_files)
import ipdb;ipdb.set_trace()

num_workers = 8 

split_lists = np.array_split(todo_files, 8)

for idx in range(num_workers):
    with open(os.path.join(DATA_ROOT, "tmp", f"beat_tts_{idx}.txt"), "w", encoding="utf-8") as f:
        for file_path in split_lists[idx]:
            f.write(file_path + "\n")



def worker(id):
    gpu_id = id % 8
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # env["PYTHONPATH"] = os.pathsep.join([os.path.join(ROOT, "src"), os.path.join(ROOT, "utils")])
    try:
        cmd = f"~/miniconda3/envs/qwen/bin/python {os.path.join(ROOT, 'datasets', 'BEAT_preprocess', 'qwen2_5omni_tts_batch.py')} --audio_path {os.path.join(DATA_ROOT, 'tmp', f'beat_tts_{id}.txt')} --suffix _qwen1"
        print(cmd)
        process = subprocess.Popen([cmd], env=env, stdout=subprocess.PIPE, text=True, shell=True)

        # 实时读取输出
        while True:
            print(id)
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())

        # 获取返回值
        return_code = process.wait()
        print("Return code:", return_code)
    except Exception as e:
        print(f"Error in worker {id}: {e}")
        return -1

threads = [Thread(target=worker, args=(i,)) for i in range(num_workers)]
for t in threads:
    t.start()
for t in threads:
    t.join() 