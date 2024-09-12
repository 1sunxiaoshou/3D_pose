import sys
from PyQt5.QtWidgets import QMainWindow,QApplication,QWidget,QFileDialog
from PyQt5 import QtCore,QtGui
from PyQt5.QtGui import QIcon,QImage,QPixmap,QKeySequence,QFont,QPainter,QPen,QColor
from PyQt5.QtWidgets import QMessageBox,QShortcut,QInputDialog
from PyQt5.QtCore import QTimer,Qt
from Ui_Forn import Ui_Form

import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import Pose_to_3D as P3D
 
 
model_path = 'face_model_5.keras'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 设置TensorFlow日志级别为ERROR

#3D姿态识别类
class PoseEstimator_Child(P3D.PoseEstimator):
    #重写绘图方法
    def plot_keypoints_2d(self, kps, size):
        width, height = size[0], size[1]
        # 创建QPixmap对象
        pixmap = QPixmap(width, height)
        pixmap.fill(Qt.white)  # 填充白色背景
         # 定义缩放比例
        scale_width, scale_height = width/28, height/28

        # 创建QPainter对象
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)  # 开启抗锯齿
        # 绘制关键点
        for i in range(kps.shape[0]):
            x, y, z = kps[i]
            painter.setPen(QPen(QColor(0, 0, 255), 10))  # 红色画笔
            painter.drawEllipse(int(y * scale_width), int(z * scale_height), 8, 8)


        # 绘制关节连接线
        parent = np.array([0, 1, 2, 3, 3, 1, 6, 7, 8, 8, 12, 15, 14, 15, 24, 24, 16, 17, 18, 24, 20, 21, 22, 0])-1
        for i in range(24):
            if parent[i] != -1:
                x1, y1, z1 = kps[i]
                x2, y2, z2 = kps[parent[i]]
                painter.setPen(QPen(QColor(0, 255, 0), 5))  # 绿色画笔
                painter.drawLine(
                    int(y1 * scale_width), int(z1 * scale_height),
                    int(y2 * scale_width), int(z2 * scale_height)
                )
        self._value = 100
        self.notify()

        # 完成绘制
        painter.end()
        return pixmap

    
        

 #主窗口
class MyMainWindow(QWidget,Ui_Form):
    def __init__(self,parent =None):
        super(MyMainWindow,self).__init__(parent)
        self.setupUi(self)
        self.Method_bindings()

        self.type_input = None #输入类型
        self.is_change = False #是否转换
        self.is_save = False   #是否保存
        self.image = None      #存储转换后的图片
        self.video = None      #存储转换后的视频
        self.file_path = None  #文件地址
        self.pose = PoseEstimator_Child("Resnet34_3inputs_448x448_20200609.onnx")

        self.pose.attach(self)  # 注册容器作为观察者
        self.progressBar.setValue(0)


                
        # 设置文字颜色为红色
        self.label.setStyleSheet("color: red;")
        # 设置文字居中对齐
        self.label.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setPointSize(18)  # 设置字体大小
        self.label.setFont(font)  # 将字体应用到 label



    #信号/槽绑定
    def Method_bindings(self):
        self.pushButton.clicked.connect(self.button)
        self.pushButton_2.clicked.connect(self.button_2)
        self.pushButton_3.clicked.connect(self.button_3)

    def update(self, value):
        self.progressBar.setValue(value)

    #载入
    def button(self):
        self.textEdit.append("选择文件")
        filters = "Image files (*.png *.jpg *.jpeg *.bmp);;Video files (*.mp4 *.avi *.mov *.mkv *.flv)"
        file, filetype = QFileDialog.getOpenFileName(self, "选取文件", "", filters)
        
        if file == "":
            self.textEdit.append("取消选择")
            print("\n取消选择")
            return
        
        self.textEdit.append(f"选择的文件为: {file}")
        self.file_path = file

        #重置状态
        self.is_change = False
        self.is_save = False
        self.progressBar.setValue(0) 

        # 判断选择的文件类型，并进行相应的处理
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            self.type_input = 'img'
            pixmap = QtGui.QPixmap(file) # 创建相应的QPixmap对象
            scaled_pixmap = pixmap.scaled(self.label.size(), Qt.KeepAspectRatio)
            self.label.clear()#清除文字
            self.label.setPixmap(scaled_pixmap) # 显示图像
            self.label.setAlignment(Qt.AlignCenter) # 图像居中

        elif file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv')):
            self.type_input = 'video'
            self.label.clear()#清除文字
            self.label.setPixmap(QPixmap()) 
            # 设置新的文字
            self.label.setText("暂不支持视频文件预览")

        else:
            print(f"{file} 不是支持的文件类型")
    
    #转换
    def button_2(self):
        if self.file_path is None:
            QMessageBox.warning(None, '警告', '请先选择一个文件！')
            self.textEdit.append('未选择文件')
            return
        elif self.is_change:
            QMessageBox.warning(None, '警告', '已转换，请勿重复转换！')
        elif self.type_input=='img':
            self.is_change = True
            self.textEdit.append('开始转换图像文件')
            #加载并转换图片方法
            self.img_change(self.file_path)

        elif self.type_input=='video':
            self.is_change = True
            self.label.clear()#清除文字
            self.label.setPixmap(QPixmap()) 
            self.textEdit.append('开始转换视频文件')
            self.label.setText('开始转换视频文件')
            #加载并转换视频方法
            self.video_change(self.file_path)

            # 设置新的文字
            self.label.setText("视频文件已自动保存")
            self.textEdit.append('视频文件已自动保存')   
             
        else:
            self.textEdit.append('异常文件')

    #保存
    def button_3(self):
        if self.type_input is None:
            QMessageBox.warning(None, '警告', '未选择文件,请先选择文件！')
            return
        elif self.is_save:
            QMessageBox.warning(None, '警告', '已保存，请勿重复保存！')
        elif self.is_change:
            if self.type_input=='img':
                self.is_save = True
                #保存图片
                file_path = 'skeleton_image.png'  # 定义文件路径
                self.image.save(file_path)  # 保存图像，格式由文件扩展名决定
                self.textEdit.append('成功保存图片')

            elif self.type_input=='video':
                self.is_save = True
                QMessageBox.warning(None, '提升', '视频文件已自动保存')  
                #保存视频     
        else:
            QMessageBox.warning(None, '警告', '未转换，请转换后再保存！')

    #图片转换
    def img_change(self,img_path):
        image = QImage(img_path)
        width = image.width()
        height = image.height()
        keypoints = self.pose.inference_image(img_path)
        pixmap = self.pose.plot_keypoints_2d(keypoints,[width,height])
        self.image = pixmap

        scaled_pixmap = pixmap.scaled(self.label.size(), Qt.KeepAspectRatio)#大小自适应
        self.label.clear()#清除文字
        self.label.setPixmap(scaled_pixmap) # 显示图像
        self.label.setAlignment(Qt.AlignCenter) # 图像居中
        
    #视频转换
    def video_change(self,video_path):
        # 打开视频文件
        output_video_path = "output_video_with_keypoints.mp4"
        self.pose.inference_video(video_path, output_video_path)
        
    

if __name__ == "__main__":
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    myWin = MyMainWindow()
    myWin.show()
    sys.exit(app.exec_())