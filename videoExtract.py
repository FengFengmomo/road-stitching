import cv2
import os


def extract_frames_opencv(video_path, output_dir, interval=1):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("无法打开视频文件，请检查路径或编解码器支持")

    frame_count = 0
    success = True
    png_count = 0

    while success:
        success, frame = cap.read()
        if not success:
            break

        # 按间隔保存帧
        if frame_count % interval == 0:
            output_path = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg")
            frame = frame[frame.shape[0]//2 : , :]
            cv2.imwrite(output_path, frame)
            png_count += 1


        frame_count += 1
        if png_count >= 30:
            break

    cap.release()
    print(f"共提取 {frame_count} 帧， 共得到{png_count}张图片，保存间隔为每 {interval} 帧")


# 使用示例
extract_frames_opencv("C:\\Users\\fengfengmomo\\Desktop\\DJI_20240915192155_0002_V.MP4", "output_frames", interval=30)  # 每30帧保存一帧