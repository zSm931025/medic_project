# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'medic_ui.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(830, 784)
        self.buttom_open = QtWidgets.QPushButton(Form)
        self.buttom_open.setGeometry(QtCore.QRect(20, 20, 130, 35))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.buttom_open.setFont(font)
        self.buttom_open.setObjectName("buttom_open")
        self.buttom_calibrate = QtWidgets.QPushButton(Form)
        self.buttom_calibrate.setGeometry(QtCore.QRect(20, 80, 130, 35))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.buttom_calibrate.setFont(font)
        self.buttom_calibrate.setObjectName("buttom_calibrate")
        self.input_exposure = QtWidgets.QLineEdit(Form)
        self.input_exposure.setGeometry(QtCore.QRect(20, 300, 131, 30))
        self.input_exposure.setObjectName("input_exposure")
        self.show_result = QtWidgets.QTextBrowser(Form)
        self.show_result.setGeometry(QtCore.QRect(20, 540, 131, 171))
        self.show_result.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.show_result.setFrameShadow(QtWidgets.QFrame.Plain)
        self.show_result.setObjectName("show_result")
        self.label = QtWidgets.QLabel(Form)
        self.label.setGeometry(QtCore.QRect(20, 260, 81, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.buttom_quit = QtWidgets.QPushButton(Form)
        self.buttom_quit.setGeometry(QtCore.QRect(20, 140, 130, 35))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.buttom_quit.setFont(font)
        self.buttom_quit.setObjectName("buttom_quit")
        self.buttom_save = QtWidgets.QPushButton(Form)
        self.buttom_save.setGeometry(QtCore.QRect(20, 350, 130, 35))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.buttom_save.setFont(font)
        self.buttom_save.setObjectName("buttom_save")
        self.show_primary_photo = QtWidgets.QLabel(Form)
        self.show_primary_photo.setGeometry(QtCore.QRect(160, 20, 640, 360))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.show_primary_photo.setFont(font)
        self.show_primary_photo.setFrameShape(QtWidgets.QFrame.Box)
        self.show_primary_photo.setFrameShadow(QtWidgets.QFrame.Plain)
        self.show_primary_photo.setObjectName("show_primary_photo")
        self.show_handled_photo = QtWidgets.QLabel(Form)
        self.show_handled_photo.setGeometry(QtCore.QRect(160, 390, 320, 320))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.show_handled_photo.setFont(font)
        self.show_handled_photo.setFrameShape(QtWidgets.QFrame.Panel)
        self.show_handled_photo.setObjectName("show_handled_photo")
        self.label_4 = QtWidgets.QLabel(Form)
        self.label_4.setGeometry(QtCore.QRect(490, 620, 70, 35))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(Form)
        self.label_5.setGeometry(QtCore.QRect(600, 620, 70, 35))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(Form)
        self.label_6.setGeometry(QtCore.QRect(700, 620, 70, 35))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.show_length = QtWidgets.QLabel(Form)
        self.show_length.setGeometry(QtCore.QRect(490, 670, 81, 31))
        self.show_length.setFrameShape(QtWidgets.QFrame.Panel)
        self.show_length.setText("")
        self.show_length.setObjectName("show_length")
        self.show_width = QtWidgets.QLabel(Form)
        self.show_width.setGeometry(QtCore.QRect(590, 670, 81, 31))
        self.show_width.setFrameShape(QtWidgets.QFrame.Panel)
        self.show_width.setText("")
        self.show_width.setObjectName("show_width")
        self.show_height = QtWidgets.QLabel(Form)
        self.show_height.setGeometry(QtCore.QRect(690, 670, 81, 31))
        self.show_height.setFrameShape(QtWidgets.QFrame.Panel)
        self.show_height.setText("")
        self.show_height.setObjectName("show_height")
        self.label_10 = QtWidgets.QLabel(Form)
        self.label_10.setGeometry(QtCore.QRect(500, 390, 72, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.label_10.setFont(font)
        self.label_10.setObjectName("label_10")
        self.show_name_dict = QtWidgets.QTextBrowser(Form)
        self.show_name_dict.setGeometry(QtCore.QRect(490, 420, 311, 101))
        self.show_name_dict.setFrameShape(QtWidgets.QFrame.Panel)
        self.show_name_dict.setObjectName("show_name_dict")
        self.label_11 = QtWidgets.QLabel(Form)
        self.label_11.setGeometry(QtCore.QRect(590, 540, 72, 15))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.label_11.setFont(font)
        self.label_11.setObjectName("label_11")
        self.label_12 = QtWidgets.QLabel(Form)
        self.label_12.setGeometry(QtCore.QRect(690, 540, 72, 15))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.label_12.setFont(font)
        self.label_12.setObjectName("label_12")
        self.show_side = QtWidgets.QLabel(Form)
        self.show_side.setGeometry(QtCore.QRect(590, 570, 81, 31))
        self.show_side.setFrameShape(QtWidgets.QFrame.Panel)
        self.show_side.setText("")
        self.show_side.setObjectName("show_side")
        self.show_num = QtWidgets.QLabel(Form)
        self.show_num.setGeometry(QtCore.QRect(690, 570, 81, 31))
        self.show_num.setFrameShape(QtWidgets.QFrame.Panel)
        self.show_num.setText("")
        self.show_num.setObjectName("show_num")
        self.label_15 = QtWidgets.QLabel(Form)
        self.label_15.setGeometry(QtCore.QRect(490, 540, 72, 15))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.label_15.setFont(font)
        self.label_15.setObjectName("label_15")
        self.show_label = QtWidgets.QLabel(Form)
        self.show_label.setGeometry(QtCore.QRect(490, 570, 81, 31))
        self.show_label.setFrameShape(QtWidgets.QFrame.Panel)
        self.show_label.setText("")
        self.show_label.setObjectName("show_label")
        self.label_17 = QtWidgets.QLabel(Form)
        self.label_17.setGeometry(QtCore.QRect(40, 720, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_17.setFont(font)
        self.label_17.setObjectName("label_17")
        self.label_18 = QtWidgets.QLabel(Form)
        self.label_18.setGeometry(QtCore.QRect(20, 500, 72, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_18.setFont(font)
        self.label_18.setObjectName("label_18")
        self.input_name = QtWidgets.QLineEdit(Form)
        self.input_name.setGeometry(QtCore.QRect(160, 720, 641, 35))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.input_name.setFont(font)
        self.input_name.setObjectName("input_name")

        self.retranslateUi(Form)
        self.buttom_open.clicked.connect(Form.open_camera)
        #self.input_exposure.textEdited['QString'].connect(Form.exposure_change)
        self.buttom_save.clicked.connect(Form.save_photo)
        self.buttom_quit.clicked.connect(Form.quit_app)
        self.buttom_calibrate.clicked.connect(Form.calibrate_change)
        #self.input_name.textEdited['QString'].connect(Form.name_change)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.buttom_open.setText(_translate("Form", "open"))
        self.buttom_calibrate.setText(_translate("Form", "calibrate"))
        self.label.setText(_translate("Form", "曝光"))
        self.buttom_quit.setText(_translate("Form", "quit"))
        self.buttom_save.setText(_translate("Form", "save"))
        self.show_primary_photo.setText(_translate("Form", "原始图片"))
        self.show_handled_photo.setText(_translate("Form", "分割后图片"))
        self.label_4.setText(_translate("Form", "length"))
        self.label_5.setText(_translate("Form", "width"))
        self.label_6.setText(_translate("Form", "height"))
        self.label_10.setText(_translate("Form", "name"))
        self.label_11.setText(_translate("Form", "side"))
        self.label_12.setText(_translate("Form", "num"))
        self.label_15.setText(_translate("Form", "label"))
        self.label_17.setText(_translate("Form", "输入药名"))
        self.label_18.setText(_translate("Form", "提示"))
