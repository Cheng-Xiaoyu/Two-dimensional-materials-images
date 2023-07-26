import sys
import os
import shutil
from PyQt5 import uic
from PyQt5.QtCore import QObject, QThread, QTimer, QDir, Qt
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QStandardItem, QImage, QPixmap, QPainter, QColor
from PyQt5.QtWidgets import QFileSystemModel, QGraphicsPixmapItem, QGraphicsScene, QAbstractItemView, QTreeView

import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

pixpath = './SelectPix/'
labelpath = './SelectPix/'


class SelectorUI:
    def __init__(self):
        # 从文件中加载UI定义
        self.pixname = None
        self.pixlabpath = None
        self.labname = None
        self.firstname = None
        self.pixpath = None
        self.labpath = None
        self.filePath = None
        self.model = None
        self.PathDataSet = None

        self.PathData = None
        self.fig = None
        self.listindex = 0
        self.figurexy = [2000, 900]
        self.alpha = 0.5

        self.ui = uic.loadUi("Selector.ui")

        self.Getitems()
        self.ui.pushButton_selectpath.clicked.connect(self.Getitems)

        self.ui.pushButton_up.clicked.connect(self.Goup)
        self.ui.pushButton_down.clicked.connect(self.Godown)
        self.ui.pushButton_chose.clicked.connect(self.PixChose)
        self.ui.pushButton_delete.clicked.connect(self.PixDelete)
        self.ui.pushButton_save.clicked.connect(self.PixChose)
        self.ui.pushButton_clear.clicked.connect(self.cleartxt)

        self.ui.treeView.clicked.connect(self.tree_clicked)

        # rootPathChanged rowsAboutToBeMoved columbsMoved
        # collapsed entered expanded pressed viewportEntered windowIconChanged SelectItems

        self.ui.treeView_lab.clicked.connect(self.tree_clicked_lab)

    def tree_change(self):
        pass
        print('******** Get ********')

        # self.ui.treeView_lab.SelectItems.connect(self.tree_clicked_lab)

    def tree_clicked(self, Qmodelidx=None):
        self.pixpath = self.model.filePath(Qmodelidx)
        self.firstname = self.model.fileName(Qmodelidx).split('.')[0]

        self.labname = self.firstname + '.png'
        self.pixlabpath = self.labpath + '/' + self.labname

        if self.labpathlists.count(self.labname) > 0:
            self.listindex = self.labpathlists.index(self.labname)
        # print(self.listindex)

        self.ui.graphicsView_img.setScene(self.viewPix(self.pixpath))
        self.ui.graphicsView_lab.setScene(self.viewPix(self.pixlabpath))

        if self.fig is not None:
            plt.close(self.fig)

        self.fig, ax = plt.subplots()
        mngr = plt.get_current_fig_manager()
        # to put it into the upper left corner for example:
        mngr.window.setGeometry(self.figurexy[0], self.figurexy[1], 520, 520)

        plt.imshow(Image.open(self.pixpath))
        plt.imshow(Image.open(self.pixlabpath), alpha=self.alpha)
        plt.show()

        self.ui.plainTextEdit.appendPlainText(self.pixpath)

        '''


'''

    def tree_clicked_lab(self, Qmodelidx=None):
        self.pixlabpath = self.model_lab.filePath(Qmodelidx)
        self.firstname = self.model_lab.fileName(Qmodelidx).split('.')[0]

        self.pixname = self.firstname + '.png'
        self.pixpath = self.filePath + '/' + self.pixname

        if self.filepathlists.count(self.pixname) > 0:
            self.listindex = self.filepathlists.index(self.pixname)
            # print(self.listindex)

        self.ui.graphicsView_img.setScene(self.viewPix(self.pixpath))
        self.ui.graphicsView_lab.setScene(self.viewPix(self.pixlabpath))

        if self.fig != None:
            plt.close(self.fig)

        self.fig, ax = plt.subplots()
        mngr = plt.get_current_fig_manager()
        # to put it into the upper left corner for example:
        mngr.window.setGeometry(self.figurexy[0], self.figurexy[1], 520, 520)

        plt.imshow(Image.open(self.pixpath))
        plt.imshow(Image.open(self.pixlabpath), alpha=self.alpha)
        plt.show()

        self.ui.plainTextEdit.appendPlainText(self.pixname)

        '''

'''

    def Getitems(self):

        self.ui.plainTextEdit.clear()
        if self.filePath != '' and self.labpath != '':
            self.filePath = self.ui.lineEdit_imgpath.text()
            self.labpath = self.ui.lineEdit_labpath.text()
            self.model = QFileSystemModel()
            self.model.setRootPath(self.ui.lineEdit_imgpath.text())
            self.ui.treeView.setModel(self.model)
            for col in range(1, 4):
                self.ui.treeView.setColumnHidden(col, True)
            self.ui.treeView.setRootIndex(self.model.index(self.ui.lineEdit_imgpath.text()))
            self.model_lab = QFileSystemModel()
            self.model_lab.setRootPath(self.ui.lineEdit_labpath.text())
            self.ui.treeView_lab.setModel(self.model_lab)
            for col in range(1, 4):
                self.ui.treeView_lab.setColumnHidden(col, True)
            self.ui.treeView_lab.setRootIndex(self.model_lab.index(self.ui.lineEdit_labpath.text()))

            self.filepathlists = os.listdir(self.filePath)
            self.labpathlists = os.listdir(self.labpath)
            self.filenum = len(self.filepathlists)


        else:
            self.ui.plainTextEdit.appendPlainText('输入路径')
        # self.ui.plainTextEdit.appendPlainText(self.PathDataSet)

    def viewPix(self, path):
        frame = QImage(path)
        pix = QPixmap.fromImage(frame)
        item = QGraphicsPixmapItem(pix)
        scene = QGraphicsScene()
        scene.addItem(item)
        return scene

    def Goup(self):
        if self.listindex > 0:
            self.listindex -= 1
            self.pixname = self.filepathlists[self.listindex]
            self.firstname = self.pixname.split('.')[0]
            self.pixpath = self.filePath + '/' + self.pixname
            self.labname = self.labpathlists[self.listindex]
            self.pixlabpath = self.labpath + '/' + self.labname

            self.ui.graphicsView_img.setScene(self.viewPix(self.pixpath))
            self.ui.graphicsView_lab.setScene(self.viewPix(self.pixlabpath))

            if self.fig != None:
                plt.close(self.fig)

            self.fig, ax = plt.subplots()
            mngr = plt.get_current_fig_manager()
            # to put it into the upper left corner for example:
            mngr.window.setGeometry(self.figurexy[0], self.figurexy[1], 520, 520)

            plt.imshow(Image.open(self.pixpath))
            plt.imshow(Image.open(self.pixlabpath), alpha=self.alpha)
            plt.show()

            self.ui.plainTextEdit.appendPlainText(self.firstname)
        else:
            self.ui.plainTextEdit.appendPlainText('开始')

    def Godown(self):
        self.filenum = len(self.filepathlists)
        if self.listindex < self.filenum-1:
            self.listindex += 1
            self.pixname = self.filepathlists[self.listindex]
            self.firstname = self.pixname.split('.')[0]
            self.pixpath = self.filePath + '/' + self.pixname
            self.labname = self.labpathlists[self.listindex]
            self.pixlabpath = self.labpath + '/' + self.labname

            self.ui.graphicsView_img.setScene(self.viewPix(self.pixpath))
            self.ui.graphicsView_lab.setScene(self.viewPix(self.pixlabpath))

            if self.fig != None:
                plt.close(self.fig)

            self.fig, ax = plt.subplots()
            mngr = plt.get_current_fig_manager()
            # to put it into the upper left corner for example:
            mngr.window.setGeometry(self.figurexy[0], self.figurexy[1], 520, 520)

            plt.imshow(Image.open(self.pixpath))
            plt.imshow(Image.open(self.pixlabpath), alpha=self.alpha)
            plt.show()
            self.ui.plainTextEdit.appendPlainText(self.firstname)
        else:
            self.ui.plainTextEdit.appendPlainText('末尾')

    def PixChose(self):
        shutil.copy(self.pixpath, self.ui.lineEdit_imgpath_copy.text() + '/' + self.firstname + '.png')
        shutil.copy(self.pixlabpath, self.ui.lineEdit_labpath_copy.text() + '/' + self.firstname + '.png')
        self.ui.plainTextEdit.appendPlainText('保存成功')
        self.ui.plainTextEdit.appendPlainText('已成功保存')
        self.ui.plainTextEdit.appendPlainText(str(len(os.listdir(self.ui.lineEdit_imgpath_copy.text()))))

    def PixDelete(self):
        try:
            os.remove(self.ui.lineEdit_imgpath_copy.text() + '/' + self.firstname + '.png')
            os.remove(self.ui.lineEdit_labpath_copy.text() + '/' + self.firstname + '.png')
            self.ui.plainTextEdit.appendPlainText('移除成功')
        except:
            self.ui.plainTextEdit.appendPlainText('已经移除')

    def cleartxt(self):
        self.ui.plainTextEdit.clear()


if __name__ == '__main__':
    # 获取UIC窗口操作权限

    app = QtWidgets.QApplication([])
    Selector = SelectorUI()
    Selector.ui.show()
    # app.exec_()
    sys.exit(app.exec_())
