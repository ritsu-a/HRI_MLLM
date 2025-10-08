import subprocess
import os

def urdf_to_mjcf(urdf_path, output_path):
    """
    使用 MuJoCo 的 compile 工具将 URDF 转换为 MJCF
    """
    # 找到 compile 工具路径
    mujoco_path = os.path.expanduser("~/.mujoco/mujoco210")
    compile_tool = os.path.join(mujoco_path, "bin", "compile")
    
    # 确保工具存在
    if not os.path.exists(compile_tool):
        raise FileNotFoundError(f"Compile tool not found at {compile_tool}")
    
    # 执行转换命令
    cmd = [compile_tool, urdf_path, output_path]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"转换成功！输出文件: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"转换失败: {e}")
        print(f"错误输出: {e.stderr}")
        return False

# 使用示例
urdf_file = "G1_inspire_hands.urdf"
output_xml = "G1_inspire_hands.xml"
urdf_to_mjcf(urdf_file, output_xml)