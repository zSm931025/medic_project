# -*- coding: utf-8 -*-
#! /bin/sh


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.Qt import *
import sys
import cv2
import pyrealsense2 as rs
import numpy as np
import time
import os
from help_file import *
import copy

name_label_dict = np.load("name_specification_manufacture_label__dict.npy",allow_pickle=True).item()
name_label_dict_key = list(name_label_dict.keys())
# name_label_dict = {"abc":1,"bcd":2,"cde":3,"efg":4}
# name_label_dict_key = list(name_label_dict.keys())



class MyThread(QThread):#线程类
    image_signal = pyqtSignal()  #刷新图片
    allwindow_signal = pyqtSignal()
    def __init__(self,main_thread):
        super(MyThread, self).__init__()
        self.pipeline = rs.pipeline()
        resolution_width = 1280  # pixels
        resolution_height = 720  # pixels
        frame_rate = 15  # fps
        self.rs_config = rs.config()
        self.rs_config.enable_stream(rs.stream.depth, resolution_width, resolution_height, rs.format.z16, frame_rate)
        self.rs_config.enable_stream(rs.stream.infrared, 1, resolution_width, resolution_height, rs.format.y8,
                                     frame_rate)
        self.rs_config.enable_stream(rs.stream.infrared, 2, resolution_width, resolution_height, rs.format.y8,
                                     frame_rate)
        self.rs_config.enable_stream(rs.stream.color, resolution_width, resolution_height, rs.format.bgr8, frame_rate)
        self.camera_state = False
        self.main_thread = main_thread

    def run(self,Trans,roi): #线程执行函数
        self.pipeline_profile = self.pipeline.start(self.rs_config)
        set_someparameter(self.pipeline,self.pipeline_profile)
        while 1:
            label_ok,self.RGB_dimension,self.whole_window,self.primary_color_image,self.primary_depth_data= get_information(self.pipeline,self.pipeline_profile,Trans,roi)
            self.allwindow_signal.emit()
            if self.main_thread.camera_parameter_change_state:
                color_sensor = self.pipeline_profile.get_device().query_sensors()[1]
                if self.main_thread.camera_parameter_exposure==0:
                    color_sensor.set_option(rs.option.enable_auto_exposure, 1)
                else:
                    color_sensor.set_option(rs.option.enable_auto_exposure, 0)
                    color_sensor.set_option(rs.option.exposure, self.main_thread.camera_parameter_exposure)
                self.main_thread.camera_parameter_change_state=False
            if label_ok:
                self.image_signal.emit()#
                self.main_thread.new_frame = True
            if  not self.camera_state:
                break
        cv2.destroyAllWindows()
        self.pipeline.stop()#
        self.terminate()

class Ui_MainWindow(QtWidgets.QWidget):

    def __init__(self, ):
        super().__init__()  # 父类的构造函数
        self.set_ui()  # 初始化程序界面
        self.side = 1;
        self.side_num = 1;
        self.camera_calib_state=False
        self.camera_state=False
        self.input_state = False
        self.parameter = None
        self.boundary = None
        self.parameter_state = False
        self.count = 1
        self.my_thread = MyThread(self)
        self.my_thread.allwindow_signal.connect(self.show_camera)
        self.my_thread.image_signal.connect(self.show_photo)
        self.path = os.getcwd()
        self.camera_parameter_change_state = False
        self.camera_parameter_exposure = 0
        self.new_frame = False
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_result)
        self.timer.start(5000)


        if not os.path.isdir("./DATA"):
            os.makedirs("./DATA")
        if not os.path.isdir("./DATA/handled_data"):
            os.makedirs("./DATA/handled_data")
        if not os.path.isdir("./DATA/primary_data"):
            os.makedirs("./DATA/primary_data")
        self.path_handled = self.path+"/DATA/handled_data"
        self.path_primary = self.path+"/DATA/primary_data"

        try:
            parameter_info = np.load("parameters.npy",allow_pickle=True)
            self.parameter = parameter_info[0]
            self.boundary = parameter_info[1]
            self.parameter_state = True
        except:
            self.result_input.setText("please calibrate the external parameter")

    '''程序界面布局'''
    def set_ui(self):

        self.__layout_main = QtWidgets.QHBoxLayout()  # 总布局
        self.__layout_fun_button = QtWidgets.QVBoxLayout()  # 按键布局
        self.__layout_data_show = QtWidgets.QVBoxLayout()  # 数据(视频)显示布局
        self.__layout_result = QtWidgets.QHBoxLayout()

        # 标记外参
        self.button_calibrate = QtWidgets.QPushButton('标记外参')
        self.button_calibrate.setFixedSize(120, 30)
        self.button_calibrate.setMinimumSize(120, 30)
        self.__layout_fun_button.addWidget(self.button_calibrate)
        self.button_calibrate.clicked.connect(self.calibrate)


        #打开相机
        self.button_open_camera = QtWidgets.QPushButton('打开相机')  #建立用于打开摄像头的按键
        self.button_open_camera.setFixedSize(120, 30)
        self.button_open_camera.setMinimumSize(120, 30)
        self.__layout_fun_button.addWidget(self.button_open_camera)
        self.button_open_camera.clicked.connect(self.button_open_camera_clicked)


        #RGB曝光
        intvalidator = QIntValidator(self)
        intvalidator.setRange(0,50000)
        self.RGB_exposure = QtWidgets.QLineEdit()#"RGB_auto_exposure:0")
        self.RGB_exposure.setPlaceholderText("RGB曝光:auto-0")
        self.RGB_exposure.setValidator(intvalidator)
        self.RGB_exposure.setFixedSize(120, 30)
        self.RGB_exposure.setMinimumSize(120, 30)
        self.__layout_fun_button.addWidget(self.RGB_exposure)
        self.RGB_exposure.editingFinished.connect(self.camera_exposrue)

        # 显示输入结果
        self.result_input = QtWidgets.QLabel()
        self.result_input.setText("输入结果")
        self.result_input.setFixedSize(120, 30)
        self.result_input.setMinimumSize(300, 30)
        self.__layout_fun_button.addWidget(self.result_input)

        #输入药名
        self.name_side = QtWidgets.QLineEdit()
        # self.name_side.setText('label_面数')
        self.name_side.setPlaceholderText("药名[规格1][规格2][厂家]_面数")
        self.name_side.setFixedSize(200, 30)
        self.completer = QCompleter(name_label_dict_key)
        self.name_side.setCompleter(self.completer)
        self.name_side.editingFinished.connect(self.input_info)
        self.__layout_fun_button.addWidget(self.name_side)

        #保存
        self.button_takePhoto = QtWidgets.QPushButton('保存')
        self.button_takePhoto.setFixedSize(120, 30)
        self.button_takePhoto.setMinimumSize(120, 30)
        self.__layout_fun_button.addWidget(self.button_takePhoto)
        self.button_takePhoto.clicked.connect(self.savePhoto)

        #退出
        self.button_close = QtWidgets.QPushButton('退出')  # 建立用于退出程序的按键
        self.button_close.setFixedSize(120, 30)
        self.button_close.setMinimumSize(120, 30)
        self.__layout_fun_button.addWidget(self.button_close)
        self.button_close.clicked.connect(self.close_app)

        # 显示切割的照片
        self.label_show_picture = QtWidgets.QLabel()  # 定义显示视频的Label
        self.label_show_picture.setFixedSize(360, 360)  # 给显示视频的Label设置大小为641x481
        self.__layout_result.addWidget(self.label_show_picture)

        #提示语
        self.label_show_result = QtWidgets.QLabel()
        self.label_show_result.setFixedSize(360,360)#fdgkjmni
        self.__layout_result.addWidget(self.label_show_result)

        self.__layout_data_show.addLayout(self.__layout_result)#

        # 显示视频帧
        self.label_show_camera = QtWidgets.QLabel()  # 定义显示视频的的帧
        self.label_show_camera.setFixedSize(1280, 720)  # 给显示视频的Label设置大小为641x481
        self.__layout_data_show.addWidget(self.label_show_camera)

        '''把某些控件加入到总布局中'''
        self.__layout_main.addLayout(self.__layout_fun_button)  # 把按键布局加入到总布局中
        self.__layout_main.addLayout(self.__layout_data_show)  # 把用于显示视频的Label加入到总布局中
        # self.result_layout.addWidget(self.label_show_result)
        '''总布局布置好后就可以把总布局作为参数传入下面函数'''
        self.setLayout(self.__layout_main)  # 到这步才会显示所有控件

    def calibrate(self):
        QApplication.processEvents()
        if  self.camera_state:  # 若相机打开
            self.result_input.setText("请先关闭相机")
            return
        self.camera_calib_state = True
        label,parameter,boundary=get_parameter_and_boundary()
        if not label:
            self.camera_calib_state = False
            self.parameter_state  = False
            self.result_input.setText("外参标记失败，请调整好重新标定")
            return
        else:
            self.result_input.setText("成功标记外参")
        parameters = [parameter,boundary]
        np.save("parameters.npy",parameters)
        self.parameter = parameter
        self.boundary  = boundary
        self.camera_calib_state = False
        self.parameter_state = True


    def camera_exposrue(self):
        self.camera_parameter_change_state=True
        self.camera_parameter_exposure=int(self.RGB_exposure.text())

    def update_result(self):
        self.result_input.setText("等待输入")


    def button_open_camera_clicked(self):
        if self.camera_calib_state:
            self.result_input.setText("fail to open camera,please retry after the calibration ")
            return
        if not self.parameter_state:
            self.result_input.setText("fail to open camera,please calibrate the external parameter")
            return
        if not self.camera_state:
            self.button_open_camera.setText('关闭相机')
            self.camera_state=True
            self.my_thread.camera_state=True
            # print(self.parameter,self.boundary)
            self.my_thread.run(self.parameter,self.boundary)

        else:
            self.button_open_camera.setText('打开相机')
            self.camera_state = False
            self.my_thread.camera_state=False
            self.label_show_camera.clear()  # 清空视频显示区域

    def show_photo(self):
        self.primary_color_image = self.my_thread.primary_color_image
        self.primary_depth_data = self.my_thread.primary_depth_data
        self.handled_color = self.my_thread.RGB_dimension[0][0]
        self.handled_dimensions = self.my_thread.RGB_dimension[0][1]

        QtWidgets.QApplication.processEvents()
        length = int(self.my_thread.RGB_dimension[0][1][0]*1000)
        width = int(self.my_thread.RGB_dimension[0][1][1]*1000)
        height = int(self.my_thread.RGB_dimension[0][1][2]*1000)
        self.label_show_result.setText("length:"+str(length)+",width:"+str(width)+",height:"+str(height))

        show1 = self.my_thread.RGB_dimension[0][0]
        show1 = cv2.resize(show1, (360,360))
        self.show1 = cv2.cvtColor(show1, cv2.COLOR_BGR2RGB)  # 视频色彩转换回RGB，这样才是现实的颜色
        showImage1 = QtGui.QImage(self.show1.data, show1.shape[1], show1.shape[0],
                                 QtGui.QImage.Format_RGB888)  # 把读取到的视频数据变成QImage形式
        self.label_show_picture.setPixmap(QtGui.QPixmap.fromImage(showImage1))  # 往显示视频的Label里 显示QImage

    def show_camera(self):
        QtWidgets.QApplication.processEvents()
        show = self.my_thread.whole_window
        show = cv2.resize(show, (1280, 720))  # 把读到的帧的大小重新设置为 640x480
        self.show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)  # 视频色彩转换回RGB，这样才是现实的颜色
        showImage = QtGui.QImage(self.show.data, show.shape[1], show.shape[0],
                                 QtGui.QImage.Format_RGB888)  # 把读取到的视频数据变成QImage形式
        self.label_show_camera.setPixmap(QtGui.QPixmap.fromImage(showImage))  # 往显示视频的Label里 显示QImage

    def input_info(self):
        str_name_side = self.name_side.text()
        if len(str_name_side)<3 or str_name_side[-2]!="_":
            self.result_input.setText("请检查你的输入，格式为，药_面数")
            self.input_state = False
            return
        name = str_name_side[:-2]
        side = str_name_side[-1]
        num_label = ['1','2','3','4','5','6']
        if not side in num_label:
            self.result_input.setText("输入错误，请重新输入具体数字")
            self.input_state = False
        else:
            side = int(side)
            self.side = side
            self.result_input.setText("输入正确")
            if not str_name_side[:-2] in name_label_dict_key:
                self.name = self.name = str_name_side[:-2]
            else:
                self.name = str(name_label_dict[str_name_side[:-2]])
            self.side_num = 1
            self.input_state = True

    def close_app(self):
        if self.camera_calib_state:
            self.result_input.setText("请等待标记完成")
            return
        if self.camera_state:
            self.result_input.setText("请先关闭相机")
            return

        reply = QMessageBox.question(self,"关闭程序？","你确认关闭程序吗",QMessageBox.Yes|QMessageBox.No,QMessageBox.No)
        if reply==QMessageBox.Yes:
            self.timer.stop()
            self.close()




    def savePhoto(self):
        if not self.camera_state:
            self.result_input.setText("你还没有打开相加啊")
            return
        if not self.input_state:
            self.result_input.setText("你的输入有误，请检查后再保存")
            return
        #path
        if not self.new_frame:
            self.result_input.setText("每有新的图片")
            return
        path0 = self.path_primary+"/name_"+self.name+"/"+"side_"+str(self.side)+"/"+"num_"+str(self.side_num)
        path1 = self.path_handled+"/name_"+self.name+"/"+"side_"+str(self.side)+"/"+"num_"+str(self.side_num)
        if not os.path.isdir(path0):
            os.makedirs(path0)
        if not os.path.isdir(path1):
            os.makedirs(path1)
        cv2.imwrite(path0+"/"+str(self.side_num)+".png",self.primary_color_image)
        np.save(path0+"/"+str(self.side_num)+".npy",self.primary_depth_data)
        cv2.imwrite(path1+"/"+str(self.side_num)+".png",self.handled_color)
        np.save(path1+"/"+str(self.side_num)+".npy",self.handled_dimensions)
        self.side_num+=1

        #
        path = os.getcwd()+"/DATA/garbage"
        if not os.path.isdir(path):
            os.makedirs(path)
        cv2.imwrite(path+"/"+str(self.count)+".png",self.primary_color_image)
        cv2.imwrite(path+"/handled"+str(self.count)+".png",self.handled_color)
        self.count+=1

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)  # 固定的，表示程序应用
    ui = Ui_MainWindow()  # 实例化Ui_MainWindow
    ui.show()  # 调用ui的show()以显示。同样show()是源于父类QtWidgets.QWidget的
    sys.exit(app.exec_())  # 不加这句，程序界面会一闪而过








