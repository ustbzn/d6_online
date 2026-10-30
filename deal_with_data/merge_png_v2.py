import os
from PIL import Image

def combine_custom_layout(input_dir, output_dir):
    """
    自定义拼接方式：
    第一行：camera-1 + camera-4
    第二行：camera-2 + camera-3
    第三行：lidar（单独居中）
    """
    # 定义需要处理的子目录结构
    # layout = [
    #     ["camera-0"],  # 第一行
    #     ["camera-5", "camera-1", "camera-6"],  # 第二行
    #     ["camera-3", "lidar", "camera-4"],  # 第二行
    #     ["camera-8", "camera-2", "camera-7"]
    # ]
    layout = [
        ["lidar"],  # 第一行
        ["camera-8", "camera-0", "camera-7"],  # 第二行
        ["camera-5", "camera-1", "camera-6"],  # 第二行
        ["camera-3", "camera-2", "camera-4"]
    ]
    # ['60_front', '120_front', '120_back', 'left_back', 'right_back', '120_left', '120_right', '60_left_front', '60_right_front']

    # 验证目录结构
    required_dirs = {d for row in layout for d in row}
    available_dirs = {d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))}
    missing_dirs = required_dirs - available_dirs
    if missing_dirs:
        raise FileNotFoundError(f"缺少必要目录：{missing_dirs}")

    # 收集所有文件名（取各目录文件的交集）
    file_sets = []
    for row in layout:
        for subdir in row:
            subdir_path = os.path.join(input_dir, subdir)
            file_sets.append(set(f for f in os.listdir(subdir_path) if f.endswith(".png")))
    common_files = set.intersection(*file_sets)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 处理每个文件
    for filename in common_files:
        rows = []
        max_row_width = 0  # 记录最大行宽用于最终居中

        # 处理每行
        for index, row_dirs in enumerate(layout):
            row_images = []
            
            # 加载该行所有图片
            for subdir in row_dirs:
                img_path = os.path.join(input_dir, subdir, filename)
                img = Image.open(img_path)

                # Resize所有图片为 1379x739
                if index == 0:
                    img_resized = img.resize((2758, 1478), Image.Resampling.LANCZOS)
                else:
                    img_resized = img.resize((1379, 739), Image.Resampling.LANCZOS)
                row_images.append(img_resized)

            # 统一行内高度（取最小高度）
            min_height = min(img.height for img in row_images)

            # 等比例缩放图片（确保图片高度为 739，宽度调整为等比例）
            resized = []
            for img in row_images:
                ratio = min_height / img.height
                new_width = int(img.width * ratio)
                resized.append(img.resize((new_width, min_height), Image.Resampling.LANCZOS))

            # 创建行拼接图
            row_width = sum(img.width for img in resized)
            row_img = Image.new("RGB", (row_width, min_height))
            
            x_offset = 0
            for img in resized:
                row_img.paste(img, (x_offset, 0))
                x_offset += img.width
            
            rows.append(row_img)
            max_row_width = max(max_row_width, row_img.width)

        # 创建最终画布（所有行纵向拼接）
        total_height = sum(img.height for img in rows)
        combined = Image.new("RGB", (max_row_width, total_height))

        # 逐行粘贴（每行居中）
        y_offset = 0
        for row_img in rows:
            x_center = (max_row_width - row_img.width) // 2
            combined.paste(row_img, (x_center, y_offset))
            y_offset += row_img.height

        # 保存结果
        combined.save(os.path.join(output_dir, filename))

if __name__ == "__main__":
    combine_custom_layout(
        input_dir="/pcpt/pcpt/project/atlb/model_online/multimodal_bevfusion/viz/bevf_cam9_lidar2_online_16k_spconv_finetuning_6class_ep50_visual_bag_test_1",
        output_dir="/pcpt/pcpt/project/atlb/model_online/multimodal_bevfusion/viz/bevf_cam9_lidar2_online_16k_spconv_finetuning_6class_ep50_visual_bag_test_1_merge"
    )
