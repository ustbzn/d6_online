import cv2
import os
import argparse



def images_to_video(image_folder, output_video_path, fps=30):
    """
    将文件夹中的图片序列转换为视频。

    :param image_folder: 包含图片的文件夹路径
    :param output_video_path: 输出视频文件路径
    :param fps: 视频帧率，默认30fps
    """
    # 获取文件夹中的所有图片文件
    images = [img for img in os.listdir(image_folder) if img.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))]
    images.sort()  # 按文件名排序

    if not images:
        print("文件夹中没有找到图片文件。")
        return

    # 读取第一张图片以获取尺寸
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用mp4v编码
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for image in images:
        image_path = os.path.join(image_folder, image)
        # frame = read_image_preserve_color(image_path)
        frame = cv2.imread(image_path)
        if frame is not None:
            video.write(frame)
        else:
            print(f"无法读取图片: {image_path}")

    video.release()
    print(f"视频已保存到: {output_video_path}")

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="将图片序列转换为视频")
    # parser.add_argument("image_folder", help="包含图片的文件夹路径")
    # parser.add_argument("output_video", help="输出视频文件路径 (例如: output.mp4)")
    # parser.add_argument("--fps", type=int, default=30, help="视频帧率 (默认: 30)")

    # args = parser.parse_args()
    image_folder = '/pcpt/pcpt/project/atlb/model_online/multimodal_bevfusion/viz/bevf_cam9_lidar2_online_16k_spconv_finetuning_6class_ep50_visual_bag_test_1_merge'
    output_video = '/pcpt/pcpt/project/atlb/model_online/multimodal_bevfusion/viz/bag.mp4'
    fps = 5
    images_to_video(image_folder, output_video, fps)