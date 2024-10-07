import os
import sys

# sys.path.append("preprocess") # 将当前目录添加到Python的模块搜索路径中
# 设置目标目录路径
target_dir = "FFplusplus"

# 遍历目标目录中的每个子目录
for root, dirs, files in os.walk(target_dir):
    for file in files:
        # 构建当前文件的完整路径
        file_path = os.path.join(root, file)

        # 获取文件所在的一级目录名称
        parent_dir = os.path.basename(os.path.dirname(file_path))

        # 构建新的文件名
        new_file_name = f"{parent_dir}_{file}"

        # 构建新的文件路径
        new_file_path = os.path.join(root, new_file_name)

        # 重命名文件
        os.rename(file_path, new_file_path)
        print(f"Renamed {file_path} to {new_file_path}")
