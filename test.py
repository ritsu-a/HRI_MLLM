import os

def find_wav_files(root_dir, output_file):
    """
    查找指定目录及其子目录中的所有WAV文件，并将路径写入文本文件
    
    Args:
        root_dir (str): 要搜索的根目录路径
        output_file (str): 输出文本文件的路径
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                
                if file.endswith('.TextGrid'):
                    full_path = os.path.join(root, file).replace(".TextGrid", ".wav")
                    f.write(full_path + '\n')
                    print(f"找到WAV文件: {full_path}")

if __name__ == "__main__":
    # 设置要搜索的根目录（可以修改为你的目标目录）
    root_directory = "/root/pengyang/codebase/HRI_MLLM/data/beat_english_v0.2.1"
    
    # 设置输出文件路径（可以修改为你想要的输出路径）
    output_txt = "all.txt"
    
    # 检查输入的目录是否存在
    if not os.path.isdir(root_directory):
        print(f"错误: 目录 '{root_directory}' 不存在!")
    else:
        find_wav_files(root_directory, output_txt)
        print(f"\n完成! 所有WAV文件路径已保存到: {output_txt}")