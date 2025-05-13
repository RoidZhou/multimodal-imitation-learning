import os


def rename_files(folder_path):
    # 获取文件夹内所有文件
    files = os.listdir(folder_path)

    # 过滤掉子文件夹，只保留文件
    files = [f for f in files if os.path.isfile(os.path.join(folder_path, f))]

    # 确保有50个文件
    if len(files) != 100:
        print(f"文件夹内文件数量不是50个，当前有{len(files)}个文件")
        return

    # 对文件进行排序（如果需要按特定顺序重命名）
    files.sort()

    # 开始重命名
    start_num = 100
    for i, filename in enumerate(files):
        # 获取文件扩展名
        # file_ext = os.path.splitext(filename)[1]

        # 新文件名
        new_name = start_num + i
        new_name = f"{new_name:.0f}{'.0.0.0'}"

        # 旧文件完整路径
        old_file = os.path.join(folder_path, filename)
        # 新文件完整路径
        new_file = os.path.join(folder_path, new_name)

        # 重命名
        os.rename(old_file, new_file)
        print(f"重命名: {filename} -> {new_name}")


# 使用示例
folder_path = "/home/zhou/autolab/imitation_learning_idp3/data/ur5_assembly/ur5_assembly_20_.zarr/data/image_hand"  # 替换为你的实际文件夹路径
rename_files(folder_path)