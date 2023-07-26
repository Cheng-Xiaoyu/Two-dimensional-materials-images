import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

from matplotlib.figure import Figure

import sys
from PyQt5 import uic
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QStandardItem, QImage, QPixmap, QPainter, QColor
from PyQt5.QtWidgets import QFileSystemModel, QGraphicsPixmapItem, QGraphicsScene



class testUI:
    def __init__(self):
        # 从文件中加载UI定义

        self.ui = uic.loadUi("untitled.ui")
        self.mask_path = r'E:\jupyter notebook\DeepLabV3Plus-Pytorch-master\test_results\img00000038.png'
        self.img_path = r'E:\jupyter notebook\DeepLabV3Plus-Pytorch-master\test_target\img00000038.png'

        self.ui.graphicsView.setScene(self.viewPix(self.img_path))

        self.ui.label.setPixmap(QPixmap.fromImage(QImage(self.img_path)))
        self.ui.label.setPixmap(QPixmap.fromImage(QImage(self.mask_path)))

        self.ui.graphicsView.setScene(self.AlphaviewPix(self.mask_path))
        self.ui.pushButton.clicked.connect(self.button)

    def button(self):
        print('button ok')
        figs = Figure()
        axes = figs.add_subplot(111)
        axes.imshow(Image.open(self.img_path))
        axes.imshow(Image.open(self.mask_path), alpha=0.7)
        axes.show()


        #plt.show()
        #scene = QGraphicsScene()
        #scene.addWidget(fig)
        #self.ui.graphicsView.setScene(scene)
        #pix=plt.imshow(Image.open(self.img_path))
        #pix=plt.imshow(Image.open(self.mask_path),alpha=0.7)




    def viewPix(self, path):
        frame = QImage(path)
        pix = QPixmap.fromImage(frame)
        item = QGraphicsPixmapItem(pix)
        scene = QGraphicsScene()
        scene.addItem(item)
        return scene

    def AlphaviewPix(self, path):
        frame = QImage(path)
        pix = QPixmap.fromImage(frame)
        item = QGraphicsPixmapItem(pix)
        scene = QGraphicsScene()
        scene.addItem(item)
        return scene


if __name__ == '__main__':

    app = QtWidgets.QApplication([])
    testor = testUI()
    testor.ui.show()
    # app.exec_()
    sys.exit(app.exec_())

'''
    # 获取UIC窗口操作权限
    mask_path=r'E:\jupyter notebook\DeepLabV3Plus-Pytorch-master\test_results\img00000038.png'
    img_path=r'E:\jupyter notebook\DeepLabV3Plus-Pytorch-master\test_target\img00000038.png'

    mask=Image.open(mask_path).convert('RGB')
    img =Image.open(img_path)
    plt.axis('off')
    plt.imshow(img)
    plt.imshow(mask, alpha=0.6)
    ax = plt.gca()
    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
    #plt.savefig('results/%d_overlay.png' % 38, bbox_inches='tight', pad_inches=0)
    plt.show()
    

    '''

