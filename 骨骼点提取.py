import os
import sys
import cv2
from sys import platform
from bin import pyopenpose as op

# 获取脚本所在目录
dir_path = os.path.dirname(os.path.realpath(__file__))
# 添加OpenPose的bin目录到系统路径
os.environ['PATH'] = os.environ['PATH'] + ';' + dir_path + '/bin;'


# 设置 OpenPose 的参数
params = {
    "model_folder": "E:\C_tool\openpose-master\models",
    "tracking": 1,# 开启追踪功能
    "number_people_max": 1 # 限制跟踪的人数为1  
}

# 初始化 OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

def process_image(image_path):
    """处理图像并返回关键点"""
    image = cv2.imread(image_path)
    datum = op.Datum()
    datum.cvInputData = image
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))
    return datum.poseKeypoints, datum.cvOutputData

def process_video(video_path):
    """处理视频流，返回关键点和处理后的视频帧"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        opWrapper.stop()
        cv2.destroyAllWindows()
        exit()

    # 获取视频属性
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))//2, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))//2)
    keypoints_list = []
    processed_frames = []

    c = 0
    while True:
        hasFrame, frame = cap.read()
        if not hasFrame:
            break

        # 根据需要缩放图像
        img_resize = cv2.resize(frame, size)  # 确保使用正确的尺寸

        datum = op.Datum()
        datum.cvInputData = img_resize
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))
        keypoints = datum.poseKeypoints
        processed_frame = datum.cvOutputData

        keypoints_list.append(keypoints)
        processed_frames.append(processed_frame)

        # 计算进度并打印进度条
        progress = c / frame_count
        sys.stdout.write(f"\rProcessing video: [{int(progress*50)*'#'}{(50 - int(progress*50))*' '}] {int(progress*100)}%")
        sys.stdout.flush()

        c += 1

    cap.release()
    return keypoints_list, processed_frames, fps



# 使用示例
if __name__ == "__main__":
    # # 处理图像
    # image_path = "down.jpeg"
    # keypoints, image_with_keypoints = process_image(image_path)
    # print(keypoints)
    # # 显示图像
    # cv2.imshow("Image with Keypoints", image_with_keypoints)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 处理视频
    video_path = "video.mp4"
    keypoints_list, processed_frames, fps= process_video(video_path)
    print('处理完毕！')

    # 获取视频的尺寸
    frame = processed_frames[0]  # 假设至少有一帧
    height, width = frame.shape[:2]

    # 定义输出视频的参数
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 定义编解码器和文件格式
    output_path = "dance_processed.mp4"  # 输出视频的路径

    # 创建VideoWriter对象
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    # 使用原始FPS来控制显示帧率
    if processed_frames:
        for frame in processed_frames:
            if frame is not None and frame.size > 0:
                cv2.imshow("Video Stream with Keypoints", frame)
                # 写入帧到输出视频
                out.write(frame)
                # 计算等待时间（ms），1000ms / fps
                wait_time = int(1000 / fps)
                # 显示帧并等待指定的时间
                if cv2.waitKey(wait_time) & 0xFF == ord('q'):
                    break
            else:
                print("Error: Frame is None or has zero size.")
                break
    # 释放资源
    out.release()
    cv2.destroyAllWindows()
    opWrapper.stop()
