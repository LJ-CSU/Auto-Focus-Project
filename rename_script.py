import os
import glob
from natsort import natsorted

def rename_images_in_folder(folder_path, prefix, start_num=1):
    """
    按顺序重命名文件夹中的所有图片

    参数:
    folder_path: 图片文件夹路径
    prefix: 重命名后的文件前缀
    start_num: 起始编号，默认为1
    """

    # 支持的图片格式
    image_extensions = ['*.png','*.bmp']

    # 获取所有图片文件
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))

    if not image_files:
        print(f"在文件夹 '{folder_path}' 中没有找到图片文件")
        return

    # 按文件修改时间排序（保持原有顺序）
    # image_files.sort(key=lambda x: os.path.getmtime(x))
    image_files = natsorted(image_files)

    print(f"找到 {len(image_files)} 个图片文件:")
    for i, file in enumerate(image_files, 1):
        print(f"{i:3d}. {os.path.basename(file)}")

    print("\n开始重命名...")

    renamed_count = 0
    for i, old_path in enumerate(image_files, start=start_num):
        # 获取文件扩展名
        _, ext = os.path.splitext(old_path)

        # 生成新文件名
        new_filename = f"{prefix}{i:02d}{ext}"  # 使用4位数字编号，如 image_001.jpg
        new_path = os.path.join(folder_path, new_filename)

        # 避免文件名冲突
        counter = 1
        while os.path.exists(new_path):
            new_filename = f"{prefix}{i:02d}_{counter}{ext}"
            new_path = os.path.join(folder_path, new_filename)
            counter += 1

        try:
            # 重命名原文件
            os.rename(old_path, new_path)
            print(f"✓ 已重命名: {os.path.basename(old_path)} -> {new_filename}")
            renamed_count += 1

        except Exception as e:
            print(f"✗ 重命名失败 {os.path.basename(old_path)}: {str(e)}")

    print(f"\n完成！成功重命名 {renamed_count} 个文件")

if __name__ == "__main__":
    folder_path = r"D:\ImageProcessing\Project\datasets\cutpic_20000\z"
    rename_images_in_folder(
        folder_path=folder_path,
        prefix="img_z",      # 新文件名前缀
        start_num=0,        # 起始编号
    )