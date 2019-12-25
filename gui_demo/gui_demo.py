from PyQt5 import QtWidgets,QtGui
from PyQt5.Qt import *
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QThread
import sys
from medic_ui import Ui_Form
from guidemo_help import *
import numpy as np
import time
import os


name_label_dict = np.load("changzhengzong_label__dict.npy",allow_pickle=True).item()
name_label_dict_key = list(name_label_dict.keys())

class child_thread(QThread):
    new_primary_photo = pyqtSignal()
    new_handeld_photo = pyqtSignal()
    def __init__(self,father):
        self.father = father
        super(child_thread,self).__init__()
        self.calibrate_state = False
        try:
            parameter_info = np.load("parameters.npy", allow_pickle=True)
            self.parameter = parameter_info[0]
            self.boundary = parameter_info[1]
            self.calibrate_state = True
            self.father.show_result.setText("you already have the camera parameters")
        except:
            self.father.show_result.setText("please calibrate the external parameter")
        self.pipeline = rs.pipeline()
        resolution_width = 1280  # pixels
        resolution_height = 720  # pixels
        color_resolution_width = 1280  # pixels
        color_resolution_height = 720  # p
        frame_rate = 15  # fps
        self.rs_config = rs.config()
        self.rs_config.enable_stream(rs.stream.depth, resolution_width, resolution_height, rs.format.z16, frame_rate)
        self.rs_config.enable_stream(rs.stream.infrared, 1, resolution_width, resolution_height, rs.format.y8,
                                     frame_rate)
        self.rs_config.enable_stream(rs.stream.infrared, 2, resolution_width, resolution_height, rs.format.y8,
                                     frame_rate)
        self.rs_config.enable_stream(rs.stream.color, color_resolution_width, color_resolution_height, rs.format.bgr8,
                                     frame_rate)

    def run(self):
        self.pipeline_profile = self.pipeline.start(self.rs_config)
        set_someparameter(self.pipeline, self.pipeline_profile)
        while self.father.camera_state:
            QApplication.processEvents()
            time.sleep(0.01)
            color_image,depth_data,frames = get_primary_photo(self.pipeline,self.pipeline_profile)
            self.primary_image = color_image.copy()
            self.primary_depth = depth_data.copy()
            if self.father.camera_calibrate:
                label, parameter, boundary = calibrate_camera(self.pipeline,self.pipeline_profile)
                print("debug0")
                if not label:

                    #self.calibrate_state = False

                    self.father.buttom_calibrate.setEnabled(True)
                    self.father.camera_calibrate = False
                    continue
                else:
                    print("debug0.5")
                    self.calibrate_state = True

                print("debug1")
                parameters = [parameter, boundary]
                np.save("parameters.npy", parameters)
                self.parameter = parameter
                self.boundary = boundary
                self.father.buttom_calibrate.setEnabled(True)
                self.father.camera_calibrate = False
                #time.sleep(0.04)
            QApplication.processEvents()
            if self.father.exposure_change_state:
                color_sensor = self.pipeline_profile.get_device().query_sensors()[1]
                if self.father.camera_exposure==0:
                    color_sensor.set_option(rs.option.enable_auto_exposure, 1)
                else:
                    color_sensor.set_option(rs.option.enable_auto_exposure, 0)
                    color_sensor.set_option(rs.option.exposure, self.father.camera_exposure)
                self.father.exposure_change_state=False


            if self.calibrate_state:
                label_ok,self.RGB_dimension,self.color_image=get_handled_photo(self.pipeline,self.pipeline_profile,
                                frames,color_image,depth_data,self.parameter,self.boundary)
                if label_ok:
                    self.new_handeld_photo.emit()
                # cv2.imshow("hah", self.color_image)
            self.new_primary_photo.emit()
        self.pipeline.stop()


class main_code(QtWidgets.QWidget,Ui_Form):
    def __init__(self):
        super(main_code,self).__init__()
        self.setupUi(self)
        intvalidator = QIntValidator(self)
        intvalidator.setRange(0, 50000)
        self.input_exposure.setPlaceholderText("RGB曝光:auto-0")
        self.input_exposure.setValidator(intvalidator)
        self.input_exposure.editingFinished.connect(self.exposure_change)

        self.input_name.setPlaceholderText("药名[规格1][规格2][厂家]_面数")
        self.completer = QCompleter(name_label_dict_key)
        self.input_name.setCompleter(self.completer)
        self.input_name.editingFinished.connect(self.name_change)

        self.camera_state = False
        self.camera_calibrate = False
        self.exposure_change_state = False
        self.input_state = False
        self.new_frame = False
        self.count = 1
        self.path = os.getcwd()
        if not os.path.isdir("./DATA"):
            os.makedirs("./DATA")
        if not os.path.isdir("./DATA/handled_data"):
            os.makedirs("./DATA/handled_data")
        if not os.path.isdir("./DATA/primary_data"):
            os.makedirs("./DATA/primary_data")
        self.path_handled = self.path + "/DATA/handled_data"
        self.path_primary = self.path + "/DATA/primary_data"

        self.thread = child_thread(self)
        self.thread.new_primary_photo.connect(self.show_camera)
        self.thread.new_handeld_photo.connect(self.show_handled)


    def open_camera(self):

        if self.camera_state == False:
            self.buttom_open.setText("close")
            self.camera_state = True
            self.thread.start()
            print("hah")
            return
        if self.camera_state == True:
            self.camera_state = False
            time.sleep(0.1)
            self.buttom_open.setText("open")
            return


    def calibrate_change(self):
        if self.camera_state==False:
            self.show_result.setText("请先打开相机才能标记！")
            return
        self.buttom_calibrate.setEnabled(False)
        self.camera_calibrate=True


    def name_change(self):
        str_name_side = self.input_name.text()
        if len(str_name_side) < 3 or str_name_side[-2] != "_":
            self.show_result.setText("请检查你的输入，格式为，药_面数")
            self.input_state = False
            return
        name = str_name_side[:-2]
        side = str_name_side[-1]
        num_label = ['1', '2', '3', '4', '5', '6']
        if not side in num_label:
            self.show_result.setText("输入错误，请重新输入具体数字")
            self.input_state = False
        else:
            self.name = name
            self.side = int(side)
            self.show_result.setText("输入正确")
            if not name in name_label_dict_key:
                self.name_label = name
            else:
                self.name_label = str(name_label_dict[name])
            self.side_num = 0
            self.input_state = True
            self.show_name_dict.setText("药名+规格:"+self.name+"\n编号："+self.name_label)
            self.show_label.setText(self.name_label)
            self.show_side.setText(str(self.side))
            self.show_num.setText(str(self.side_num))
            self.input_name.clearFocus()
            self.buttom_save.setFocus()


    def exposure_change(self):
        self.exposure_change_state = True
        self.camera_exposure = int(self.input_exposure.text())
        self.input_exposure.clearFocus()

    def quit_app(self):
        if self.camera_state:
            self.show_result.setText("请先关闭相机")
            return
        reply = QMessageBox.question(self,"关闭程序？","你确认关闭程序吗",QMessageBox.Yes|QMessageBox.No,QMessageBox.No)
        if reply==QMessageBox.Yes:
            self.close()

    def save_photo(self):
        print("debug3")
        if not self.camera_state:
            self.show_result.setText("你还没有打开相加啊!")
            return
        if not self.input_state:
            self.show_result.setText("你的输入有误，请检查后再保存")
            return
        if not self.new_frame:
            self.show_result.setText("重复保存，请等待下一帧")
            return
        #path

        self.side_num += 1
        path0 = self.path_primary+"/label_"+self.name_label+"/"+"side_"+str(self.side)+"/"+"num_"+str(self.side_num)
        path1 = self.path_handled+"/label_"+self.name_label+"/"+"side_"+str(self.side)+"/"+"num_"+str(self.side_num)
        if not os.path.isdir(path0):
            os.makedirs(path0)
        if not os.path.isdir(path1):
            os.makedirs(path1)
        #ubuntu
        # cv2.imwrite(path0+"/"+self.name_label+"_"+str(self.side)+"_"+str(self.side_num)+"_primary.png",self.primary_color_image)
        #win10
        cv2.imencode('.png', self.primary_color_image)[1].tofile(path0+"/"+self.name_label+"_"+str(self.side)+"_"+str(self.side_num)+"_primary.png")
        np.save(path0+"/"+self.name_label+"_"+str(self.side)+"_"+str(self.side_num)+"_primary.npy",self.primary_depth_data)
        # cv2.imwrite(path1+"/"+self.name_label+"_"+str(self.side)+"_"+str(self.side_num)+"_handled.png",self.handled_color)
        cv2.imencode('.png', self.handled_color)[1].tofile(path1 + "/" + self.name_label + "_" + str(self.side) + "_" + str(self.side_num) + "_handled.png")
        np.save(path1+"/"+self.name_label+"_"+str(self.side)+"_"+str(self.side_num)+"_dimension.npy",self.handled_dimensions)
        self.show_num.setText(str(self.side_num))
        print("debug1")
        path = os.getcwd()+"/DATA/garbage"
        if not os.path.isdir(path):
            os.makedirs(path)
        # cv2.imwrite(path+"/"+self.name_label+"_"+str(self.side)+"_"+str(self.side_num)+".png",self.primary_color_image)
        cv2.imencode('.png', self.primary_color_image)[1].tofile(
            path+"/"+self.name_label+"_"+str(self.side)+"_"+str(self.side_num)+".png")
        # cv2.imwrite(path+"/handled"+self.name_label+"_"+str(self.side)+"_"+str(self.side_num)+".png",self.handled_color)
        cv2.imencode('.png', self.handled_color)[1].tofile(
            path+"/handled"+self.name_label+"_"+str(self.side)+"_"+str(self.side_num)+".png")
        self.count+=1
        self.new_frame = False
        self.show_result.setText("保存成功")

    def show_camera(self):
        # QtWidgets.QApplication.processEvents()
        show = self.thread.color_image
        show = cv2.resize(show, (640, 360))  # 把读到的帧的大小重新设置为 640x480
        self.show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)  # 视频色彩转换回RGB，这样才是现实的颜色
        showImage = QtGui.QImage(self.show.data, show.shape[1], show.shape[0],
                                 QtGui.QImage.Format_RGB888)  # 把读取到的视频数据变成QImage形式
        self.show_primary_photo.setPixmap(QtGui.QPixmap.fromImage(showImage))  # 往显示视频的Label里 显示QImage

    def show_handled(self):
        self.new_frame = True
        self.primary_color_image = self.thread.primary_image.copy()
        self.primary_depth_data = self.thread.primary_depth.copy()
        self.handled_color = self.thread.RGB_dimension[0][0].copy()
        self.handled_dimensions = self.thread.RGB_dimension[0][1].copy()

        self.show_length.setText(str(int(self.handled_dimensions[0]*1000)))
        self.show_width.setText(str(int(self.handled_dimensions[1]*1000)))
        self.show_height.setText(str(int(self.handled_dimensions[2]*1000)))

        show1 = self.handled_color.copy()
        show1 = cv2.resize(show1, (320, 320))
        self.show1 = cv2.cvtColor(show1, cv2.COLOR_BGR2RGB)  # 视频色彩转换回RGB，这样才是现实的颜色
        showImage1 = QtGui.QImage(self.show1.data, show1.shape[1], show1.shape[0],
                                  QtGui.QImage.Format_RGB888)  # 把读取到的视频数据变成QImage形式
        self.show_handled_photo.setPixmap(QtGui.QPixmap.fromImage(showImage1))  # 往显示视频的Label里 显示QImage



if __name__=="__main__":
    app = QApplication(sys.argv)
    mc = main_code()
    mc.show()
    sys.exit(app.exec_())
