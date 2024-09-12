import numpy as np
import onnxruntime as ort
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
ort.set_default_logger_severity(3)  # 3 代表 ERROR

class PoseEstimator:
    def __init__(self, model_path):
        self.ort_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = self.load_model(model_path)
        self.inputs = self.session.get_inputs()
        self._observers = []
        self._value = 0
        self.frame_count = None
    
    def attach(self, observer):
        self._observers.append(observer)

    def notify(self):
        for observer in self._observers:
            observer.update(self._value)

        
    #读取模型
    def load_model(self, model_path):
        sess = ort.InferenceSession(model_path, providers=self.ort_providers)
        return sess

    #图像预处理
    def preprocess_image(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (448, 448))
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 1, 0)
        img = img[np.newaxis, ...]
        return img

    #图片关节提取
    def inference_image(self, img_path):
        if type(img_path) == str:
            img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
        else:
            img = img_path
        preprocessed_img = self.preprocess_image(img)
        # print('输入大小：',preprocessed_img.shape)
        pred_onx = self.session.run(None,{
        self.inputs[0].name:preprocessed_img,
        self.inputs[1].name:preprocessed_img,
        self.inputs[2].name:preprocessed_img
        })
        offset3D, heatMap3D = np.squeeze(pred_onx[2]), np.squeeze(pred_onx[3])
        keypoints = self.get_keypoints(heatMap3D, np.squeeze(offset3D))
        return keypoints

    
    #计算所有关节的位置
    def get_keypoints(self, heatMap3D, offset3D):
        num_joints = heatMap3D.shape[0] // 28
        kps = np.zeros((num_joints, 3))
        for j in range(num_joints):
            kps[j] = self.xz_pose(heatMap3D, offset3D, j)
        return kps

    #计算单个关节坐标
    def xz_pose(self, heatMap3D, offset3D, j):
        joint_heat = heatMap3D[j*28:(j+1)*28, ...]
        x, y, z = np.unravel_index(np.argmax(joint_heat), joint_heat.shape)
        pos_x = offset3D[j*28+x, y, z] + x
        pos_y = offset3D[24*28+j*28+x, y, z] + y
        pos_z = offset3D[24*28*2+j*28+x, y, z] + z
        return [pos_x, pos_y, pos_z]

    #使用 Matplotlib 绘制一个3D图
    def plot_keypoints_3d(self, kps):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        parent = np.array([
        0, 1, 2, 3, 3,  # 头部和颈部
        1, 6, 7, 8, 8,   # 躯干和脊柱
        12, 15, 14, 15, 24,  # 左臂
        24, 16, 17, 18,  # 左前臂和手
        24, 20, 21, 22,  # 右臂
        0  # 根关节（通常为髋关节或脚跟）
        ])-1
        ax.scatter3D(kps[:, 0], -kps[:, 1], -kps[:, 2], 'red')
        for i in range(24):
            if parent[i] != -1:
                ax.plot3D(
                    [kps[i, 0], kps[parent[i], 0]],
                    [-kps[i, 1], -kps[parent[i], 1]],
                    [-kps[i, 2], -kps[parent[i], 2]],
                    'gray'
                )
        # 设置坐标轴的刻度标签大小
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.set_box_aspect([kps[:, 0].ptp(), kps[:, 1].ptp(), kps[:, 2].ptp()])
        # 设置视角
        ax.view_init(elev=10, azim=180)
        plt.show()

    def draw_keypoints_on_frame(self, frame, keypoints):
        parent = np.array([0, 1, 2, 3, 3,1, 6, 7, 8, 8,12, 15, 14, 15, 24,24, 16, 17, 18,24, 20, 21, 22,0])-1
        height, width, _ = frame.shape  # 获取图像的尺寸
        scale_width, scale_height = width/28, height/28  # 定义缩放比例

        for i in range(keypoints.shape[0]):
            x, y, z = keypoints[i]
            cv2.circle(frame, (int(y * scale_width), int(z * scale_height)), 10, (0, 0, 255), -1)  # 红色圆圈

        # 绘制关节连接线
        for i in range(24):
            if parent[i] != -1:
                # 应用相同的坐标变换
                x1, y1, z1 = keypoints[i]
                y1_scaled = y1 * scale_width
                z1_scaled = z1 * scale_height  # 使用z坐标代替y坐标进行垂直位置的缩放

                p1 = (int(y1_scaled), int(z1_scaled))

                x2, y2, z2 = keypoints[parent[i]]
                y2_scaled = y2 * scale_width
                z2_scaled = z2 * scale_height  # 使用z坐标代替y坐标进行垂直位置的缩放

                p2 = (int(y2_scaled), int(z2_scaled))

                cv2.line(frame, p1, p2, (0, 255, 0),5)  # 绿色线条

        # # 调整图像尺寸以适应屏幕
        # resized_frame = cv2.resize(frame, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_LINEAR)
        # cv2.imshow('Frame with Keypoints and Connections', resized_frame)
        # cv2.waitKey(10)

        return frame

    def inference_video(self, video_path, output_video_path):
        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        codec = cv2.VideoWriter_fourcc(*'mp4v')  # 可以选择其他编解码器

        out = cv2.VideoWriter(output_video_path, codec, fps, (frame_width, frame_height))
        self._value = 0
        self.current_frame = 0
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            keypoints = self.inference_image(frame)  # 假设 inference_image 可以处理单帧
            processed_frame = self.draw_keypoints_on_frame(frame, keypoints)
            out.write(processed_frame)

            self.current_frame += 1  # 每处理一帧，当前帧数递增
            self._value = int((self.current_frame / self.frame_count) * 100) # 计算进度百分比
            self.notify()

        cap.release()
        out.release()

        return output_video_path

if __name__ == "__main__":

    # 使用方法
    model_path = "Resnet34_3inputs_448x448_20200609.onnx"
    pose_estimator = PoseEstimator(model_path)

    # 图片推理
    # img_path = 'kunkun.jpg'
    # keypoints = pose_estimator.inference_image(img_path)
    # pose_estimator.plot_keypoints_3d(keypoints)

    # 视频推理
    video_path = "video.mp4"
    output_video_path = "output_video_with_keypoints.mp4"
    pose_estimator.inference_video(video_path, output_video_path)
    # 这里可以添加代码来绘制视频中每一帧的关节点