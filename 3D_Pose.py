import numpy as np
import onnxruntime as ort
import cv2

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

ort.set_default_logger_severity(3)  # 3 代表 ERROR

providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']  # GPU优先，如果GPU不可用则回退到CPU
#加载模型
sess = ort.InferenceSession("Resnet34_3inputs_448x448_20200609.onnx",providers=providers)
inputs = sess.get_inputs()
#读取图片
img = cv2.imread("down.jpeg")
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img = cv2.resize(img,(448,448))
img = img.astype(np.float32)/255.0
img = img.transpose(2,1,0)
img = img[np.newaxis,...]
print('输入大小：',img.shape)
#输入到网络结构中
pred_onx = sess.run(None,{
    inputs[0].name:img,
    inputs[1].name:img,
    inputs[2].name:img
})

offset3D = np.squeeze(pred_onx[2])
heatMap3D = np.squeeze(pred_onx[3])
print(offset3D.shape)#(2016, 28, 28)
print(heatMap3D.shape)#(672, 28, 28)
print(offset3D.shape[0]/heatMap3D.shape[0])#3.0

def xz_pose(heatMap3D,offset3D,j):
    # 找到第j个关节的28个特征图，并找到最大值的索引
    joint_heat = heatMap3D[j*28:(j+1)*28,...]
    [x,y,z] = np.where(joint_heat==np.max(joint_heat))
    # 避免有多个最大值，所以取最后一组
    x=int(x[-1])
    y=int(y[-1])
    z=int(z[-1])

    pos_x = offset3D[j*28+x,y,z] + x
    pos_y = offset3D[24*28+j*28+x,y,z] + y
    pos_z = offset3D[24*28*2+j*28+x,y,z] + z

    return [pos_x,pos_y,pos_z]

# print(xz_pose(heatMap3D,offset3D,0))

# 获取所有关节的位置
def get_keypoints(heatMap3D, offset3D):
    num_joints = heatMap3D.shape[0] // 28
    kps = np.zeros((num_joints, 3))
    for j in range(num_joints):
        kps[j] = xz_pose(heatMap3D, offset3D, j)
    return kps

# 调用函数获取关节位置
kps = get_keypoints(heatMap3D, offset3D)
print(kps)
# print(kps)
# 定义关节的父子关系，-1 表示没有父关节
parent = np.array([
    0, 1, 2, 3, 3,  # 头部和颈部
    1, 6, 7, 8, 8,   # 躯干和脊柱
    12, 15, 14, 15, 24,  # 左臂
    24, 16, 17, 18,  # 左前臂和手
    24, 20, 21, 22,  # 右臂
    0  # 根关节（通常为髋关节或脚跟）
])-1

# 创建一个新的figure对象，并获取一个3D坐标轴
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制24个关节的坐标点
ax.scatter3D(kps[:, 0], -kps[:, 1], -kps[:, 2], 'red')

# 绘制连接关节的线条
for i in range(24):
    if parent[i] != -1:  # 如果关节i有父关节
        ax.plot3D(
            [kps[i, 0], kps[parent[i], 0]],
            [-kps[i, 1], -kps[parent[i], 1]],
            [-kps[i, 2], -kps[parent[i], 2]],
            'gray'
        )

# 设置坐标轴的刻度标签大小
ax.tick_params(axis='both', which='major', labelsize=10)

# 设置视角
ax.view_init(elev=10, azim=180)

# 显示图表
plt.show()

