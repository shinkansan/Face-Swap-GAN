#!/bin/env/python3

from ast import Or
from PyQt5.QtCore import QDir, Qt, QUrl
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QLabel,
        QPushButton, QSizePolicy, QSlider, QStyle, QVBoxLayout, QWidget)
from PyQt5.QtWidgets import QMainWindow,QWidget, QPushButton, QAction
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import *
import PyQt5.QtWidgets as QtWidgets
from PyQt5.QtCore import *
from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import Qt
import sys, os, glob
sys.path.append("./elba_FSGAN")
from PyQt5 import uic
import numpy as np
import cv2
UIf = uic.loadUiType("./gui/gui.ui")[0]
from fd_util.fd_crop import fd_detect
from gui_adapter import engineAdapter
from PIL import Image
from multiprocessing import Process, Queue
import multiprocessing as mp
from collections import OrderedDict

FD_CROP_SIZE = 224

class specificImageHolder():

    data = OrderedDict()

    def __init__(self):
        pass

    def register(self, faceid, src, dst = None):
        self.data.update({faceid : [src, dst]})

    def remove(self):
        files = glob.glob('./elba_FSGAN/face_proc/onetoone/*')
        for f in files:
            if os.path.isfile(f): os.remove(f)

    def save(self):
        self.remove()
        for idx, img in self.data.items():
            if type(img[1]) == type(None):
                QMessageBox.warning(myWindow, "Can't Start", f"Assertion Failed : {idx} SRC and DST Faces are not matched!")
                return -1
                #raise Exception("SRC and DST are not prepared")
            SRC_FILE_NAME = os.path.join('./elba_FSGAN/face_proc/onetoone/', f"SRC_{idx}.png")
            DST_FILE_NAME = os.path.join('./elba_FSGAN/face_proc/onetoone/', f"DST_{idx}.png")
            cv2.imwrite(SRC_FILE_NAME, img[0])
            cv2.imwrite(DST_FILE_NAME, img[1])
        return 1

    def delItem(self, faceID):
        if self.data.get(faceID):
            del self.data[faceID]
            tempData = OrderedDict()

            for idx, curItem in enumerate(self.data.items()):
                tempData.update({idx : curItem[1]})

            self.data = tempData

        else:
            raise("Internal Error")


class MyWindow(QMainWindow, UIf):

    def __init__(self):
        super(QMainWindow, self).__init__()
        self.setupUi(self)

        self.pushButton_2.clicked.connect(self.open_video)
        self.pushButton.clicked.connect(self.faceDetect)

        self.player = customVideoPlayer()

        self.pushButton_3.clicked.connect(self.playStateChage)
        self.pushButton_6.clicked.connect(self.player.stop)
        self.pushButton_5.clicked.connect(lambda : self.tableWidget.setRowCount(0))
        self.horizontalSlider.sliderPressed.connect(lambda : self.player.pause())
        self.horizontalSlider.sliderReleased.connect(lambda : self.player.setVideoPos(self.horizontalSlider.value()))

        self.player.imageSig.connect(self.srcdisplay)
        self.player.tsSig.connect(self.horizontalSlider.setSliderPosition)
        self.srcCurImage = 0
        self.label_3.mousePressEvent = self.getSrcPos
        self.tableWidget.cellClicked.connect(self.registerDST)

        self.commandLinkButton.clicked.connect(self.inference)

        self.faceHolder = specificImageHolder()

        self.result_manager_timer = QTimer(self)
        self.result_manager_timer.interval = 1000
        self.result_manager_timer.timeout.connect(self.resultManager)

        self.progressBar.setValue(0)

        #self.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        self.isVideoLoad = False
        self.inferFrame = 0
        self._loadUIUpdate()

        # TODO : HQ Model apply?
        # if crop_size == 512:
        #     opt.which_epoch = 550000
        #     opt.name = '512'
        #     mode = 'ffhq'
        # else:
        #     mode = 'None'
        mode = 'None'

        self.fdInst = fd_detect()
        
        #self.fdInst.prepare(det_size=(320,320))

    def infer_test(self):

        srcVideoPath = self.video_name
        if self.outputSave == '':
            t_name, t_ext = os.path.splitext(self.video_name)
            outputVideoPath = t_name + "_mod" + t_ext
        else:
            outputVideoPath = self.outputSave

        print(outputVideoPath, srcVideoPath)
        
        ie = engineAdapter(self.fdInst,srcVideoPath, outputVideoPath)
        self.inferOutput = Queue()
        self.inferProgress = Queue()

        self.ie_proc = Process(target=ie.run, args=(self.inferOutput, self.inferProgress))
        self.ie_proc.start()

        self.result_manager_timer.start()
        
        pass


    def inference(self):
        if len(self.faceHolder.data.keys()) > 0:
            if self.faceHolder.save() == -1:
                return
        else:
            QMessageBox.warning(self, "Can't Start", "No face to change")
            return
        self.outputSave = QFileDialog.getSaveFileName(self, 'output save location', "", "MP4 (*.mp4)")[0]
        if QMessageBox.question(self, "Job Prepared!", "Do you want Proceed?\n this can not be undone!") == QMessageBox.Yes:
            self.infer_test()




    def resultManager(self):
        if self.inferOutput and self.inferProgress:
            if not self.inferOutput.empty(): 
                self.inferFrame = self.inferOutput.get_nowait()
                self.tgtdisplay(self.inferFrame)
            if not self.inferProgress.empty():
                progress = self.inferProgress.get_nowait()
                self.progressBar.setMaximum(progress[1]-1)
                self.progressBar.setValue(progress[0])
                if self.checkBox.isChecked():
                    self.player.setVideoPosandPause(progress[0])


                if progress[0] == progress[1]-1:
                    QMessageBox.information(self, "Job Done!", "It will take a time for result video showing up in the folder, (Saving Progress in backgroun!)")
                    self.result_manager_timer.stop()




                

    def registerDST(self, row, col):
        if col == 3: # Click to Delete
            # for rowIdx in range(row, self.tableWidget.rowCount()-1):
            #     prevIdx = rowIdx
            #     newIdx = rowIdx - 1
            #     if newIdx < 0 : raise Exception("WHAT???????")

            #     self.faceHolder.data[newIdx] = self.faceHolder.data[prevIdx]

            # del self.faceHolder.data[self.tableWidget.rowCount() - 1]
            self.faceHolder.delItem(row)
            self.tableWidget.removeRow(row)


        if col != 2:
            return
        img_name, _ = QFileDialog.getOpenFileName(self, "Select DST Image",
                QDir.homePath())
        if img_name == '': return
        dst_img = cv2.imread(img_name)

        faceImg = np.asarray(cv2.cvtColor(dst_img,cv2.COLOR_BGR2RGB))   
        width, height,_ = faceImg.shape
        msg = faceImg.data
        qimg = QImage(msg, height, width, QImage.Format_RGB888)
        pic = QPixmap(qimg)
        pic = pic.scaled(100, 100, Qt.KeepAspectRatio)
        self.tempLabel = QtWidgets.QLabel("FaceImg")
        self.tempLabel.setPixmap(pic)
        self.tableWidget.setCellWidget(row, 2, self.tempLabel)

        if self.faceHolder.data.get(row):
            self.faceHolder.register(row,self.faceHolder.data[row][0], dst_img )

    def getSrcPos(self, e):
        x = e.x()
        y = e.y() 
        print(x, y)

    def _loadUIUpdate(self):

        if self.isVideoLoad:
            self.pushButton_6.setEnabled(True)
            self.pushButton_3.setEnabled(True)
            self.pushButton.setEnabled(True)
            self.horizontalSlider.setEnabled(True)
            self.horizontalSlider.setSliderPosition(0)
            self.horizontalSlider.setRange(0, self.player.tot_frame)
        else:
            self.pushButton_6.setDisabled(True) 
            self.pushButton_3.setDisabled(True) 
            self.pushButton.setDisabled(True) 
            self.horizontalSlider.setDisabled(True) 
            self.horizontalSlider.setSliderPosition(0)

    def faceDetect(self, frame=False):
        #print(self.srcCurImage, frame)
        if frame == False:
            frame = self.srcCurImage
            
        else:
            raise Exception("No Image Feed for faceDetect!")

        try:
            specific_person_align_crop, _ = self.fdInst.get(frame,FD_CROP_SIZE)
            if len(specific_person_align_crop) == 0:
                print("No face in the frame!")
                return None
        except:
            print("No face image in frame")  
            QMessageBox.information(self, "No Face in frame!", "NO FACE FOUNDED") 
            return None

        for face in specific_person_align_crop:
            
            faceImg = np.asarray(Image.fromarray(cv2.resize(cv2.cvtColor(face,cv2.COLOR_BGR2RGB), dsize=(640, 640)))) 
            width, height,_ = faceImg.shape
            msg = faceImg.data
            qimg = QImage(msg, width, height, QImage.Format_RGB888)
            pic = QPixmap(qimg)
            pic = pic.scaled(100, 100, Qt.KeepAspectRatio)
            self.tempLabel = QtWidgets.QLabel("FaceImg")
            self.tempLabel.setPixmap(pic)
        
            self.tableWidget.insertRow(self.tableWidget.rowCount())
            self.faceHolder.register(self.tableWidget.rowCount() - 1, face)
            self.tableWidget.setCellWidget(self.tableWidget.rowCount() - 1, 0, self.tempLabel)
            self.tableWidget.setCellWidget(self.tableWidget.rowCount() - 1, 1, QLabel("➡"))
            self.tableWidget.setCellWidget(self.tableWidget.rowCount() - 1, 2, QLabel("Click to Set DST Face"))
            self.tableWidget.setCellWidget(self.tableWidget.rowCount() - 1, 3, QLabel("Click to Delete"))
            self.tableWidget.resizeRowsToContents()
            self.tableWidget.resizeColumnsToContents()

    # legacy - this for main screen
    @pyqtSlot(np.ndarray)
    def srcdisplay(self, image):
        #opencv
        try:
            if image is not None:
                self.srcCurImage = image.copy()
                Image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
                height, width, channel = Image.shape
                msg = Image
                qimg = QImage(msg.data, width, height, QImage.Format_RGB888)
                pix = QPixmap(qimg)
                pix = pix.scaled(511, 331, Qt.KeepAspectRatio)
                self.label_3.setPixmap(pix)

        except Exception as ex:
            print("ERROR", ex)

    def tgtdisplay(self, image):
        #opencv
        try:
            if image is not None:
                Image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
                height, width, channel = Image.shape
                msg = Image
                qimg = QImage(msg.data, width, height, QImage.Format_RGB888)
                pix = QPixmap(qimg)
                pix = pix.scaled(511, 331, Qt.KeepAspectRatio)
                self.label_4.setPixmap(pix)

        except Exception as ex:
            print("ERROR", ex)

    def playStateChage(self):
        self.player.changeState()


    def open_video(self):
        """ Slot function:
        Open a video from the file system.
        """
        
        video_name, _ = QFileDialog.getOpenFileName(self, "Open Video",
                QDir.homePath())
        self.video_name = video_name

        if self.video_name != '':
            self.player.init(self.video_name)
            self.isVideoLoad = True
            self._loadUIUpdate()
            self.player.start()

class customVideoPlayer(QThread):
    imageSig = pyqtSignal(np.ndarray)
    tsSig = pyqtSignal(int)

    def __init__(self) -> None:
        super(QThread, self).__init__()

    def init(self, videoSrc):
        self.cap = cv2.VideoCapture(videoSrc)
        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) # 또는 cap.get(3)
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # 또는 cap.get(4)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) # 또는 cap.get(5)
        self.pos_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        self.tot_frame= int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        #self.videoLoad()

        self.state = 1 # 0 for pause, 1 for play 

    def getPos(self):
        self.pos_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        return self.pos_frame

    def run(self):
        while(self.cap.isOpened()):
            if self.state and self.getPos() != self.tot_frame:
                ret, img = self.cap.read()
                self.imageSig.emit(img)
                self.tsSig.emit(self.getPos())
                self.msleep(self.fps)


    def changeState(self):
        self.state = not self.state

    def pause(self):
        self.state = 0
    def play(self):
        self.state = 1

    def stop(self):
        self.setVideoPos(0)

    def setVideoPos(self, frame):
        self.pause()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ret, img = self.cap.read()
        self.imageSig.emit(img)

        self.play()

    def setVideoPosandPause(self, frame):
        self.pause()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ret, img = self.cap.read()
        self.imageSig.emit(img)


if __name__ == "__main__":
    try:
        mp.set_start_method('spawn') 
        app = QApplication(sys.argv)
        myWindow = MyWindow()
        myWindow.show()
        app.exec_()
    except Exception as ex:
        print(ex)
        QMessageBox.information(myWindow, "Severe Error Occured", "Contact Shin\n" + str(ex))
        

