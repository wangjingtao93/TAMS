import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QListWidget, QStackedWidget, QGroupBox, QFileDialog, QPushButton, QStyledItemDelegate,
    QGraphicsOpacityEffect
)
from PyQt6.QtCore import Qt, QPoint, QRect, QSize
from PyQt6.QtGui import QColor, QPixmap, QPainter, QPen, QImage
import os

import torch
from PIL import Image
import functools


# 先将第27行的self.ParaName改为要显示的参数名，然后将45行的output改为计算后输出的参数，顺序和self.ParaName对应.需要创建几行，第54行参数就传入几

class SecondLabel(QLabel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.original_pixmap = None  # 用来保存原始图像

    def set_original_pixmap(self, pixmap):
        self.original_pixmap = pixma
        self.setPixmap(pixmap if pixmap else QPixmap())


class ImageProcessingWidget(QWidget):
    def __init__(self, num):
        super().__init__()
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        # 创建主布局
        self.main_layout = QVBoxLayout(self)
        self.label_list = []
        self.GroupBoxList = []
        self.ImgLabelList = []
        for i in range(num):
            main_layout = QVBoxLayout()
            # 创建顶部布局
            top_layout = QHBoxLayout()
            group_boxes = []
            img_labels = []  # 保存标签引用，以便之后更新图片
            for name in ["enface", "生成彩照", "真实彩照"]:
                group_box = QGroupBox(name)
                group_box.setStyleSheet("""
                    QGroupBox {
                        color: white;
                        margin-top: 50px;
                        border: 1px solid silver;
                        border-radius: 5px;
                        font-size: 20px;
                        padding-top: 20px;
                    }
                    QGroupBox::title {
                        subcontrol-origin: margin;
                        subcontrol-position: top center;
                        padding: 0 3px;
                    }
                """)
                img_label = SecondLabel()
                img_label.setStyleSheet("background-color: (255, 255, 255, 255)")  # 设置标签背景为黑色
                img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)  # 设置标签内的图片居中对齐
                img_label.setMinimumSize(1, 1)

                vbox = QVBoxLayout()
                vbox.addWidget(img_label, alignment=Qt.AlignmentFlag.AlignCenter)
                group_box.setLayout(vbox)

                group_boxes.append(group_box)
                img_labels.append(img_label)  # 将标签添加到列表中
                top_layout.addWidget(group_box, 1)

            # 创建底部布局
            button_layout = QHBoxLayout()
            select_image_button = QPushButton("选择图像路径")
            select_image_button.setStyleSheet(
                "QPushButton {background-color:black;color: white;font-size: 20px }")  # 设置按钮文本颜色为白色
            select_image_button.clicked.connect(functools.partial(self.open_image_file, num=i))  # 连接按钮点击信号到槽函数
            button_layout.addWidget(select_image_button, alignment=Qt.AlignmentFlag.AlignCenter)
            self.ParaName = ["误差值"]

            label_list = []
            for name in self.ParaName:
                label = QLabel(name + ' ' + ':')
                label.setStyleSheet("QLabel {color: white; font-size: 16px;}")
                button_layout.addWidget(label, alignment=Qt.AlignmentFlag.AlignCenter)
                label_list.append(label)

            main_layout.addLayout(top_layout)
            main_layout.addLayout(button_layout)
            self.main_layout.addLayout(main_layout)
            self.label_list.append(label_list)
            self.GroupBoxList.append(group_boxes)
            self.ImgLabelList.append(img_labels)

    def open_image_file(self, num):
        file_path = QFileDialog.getExistingDirectory(self, "选择文件夹", "/home")
        if file_path:
            # output = your_function(file_path)
            output = []
            output.append(20)
            img = Image.open('C:\\Users\\Alteria\\Desktop\\img\\00_1544_L1.png')
            output.append(img)
            output.append(img)
            output.append(img)
            for i, x in enumerate(zip(self.ParaName, [output[0]])):
                self.label_list[num][i].setText(f"{x[0]} : {x[1]}")

            for i in range(0, len(output) - 1):
                self.ImgLabelList[num][i].set_original_pixmap(self.pil_image_to_pixmap(output[i + 1]))
                scaled_pixmap = self.ImgLabelList[num][i].original_pixmap.scaled(self.GroupBoxList[num][i].size(),
                                                                                 Qt.AspectRatioMode.KeepAspectRatio,
                                                                                 Qt.TransformationMode.SmoothTransformation)
                self.ImgLabelList[num][i].setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        for i in range(len(self.ImgLabelList)):
            for j in range(len(self.ImgLabelList[i])):

                if self.ImgLabelList[i][j].original_pixmap:
                    # 使用组框的尺寸来获取缩放尺寸
                    group_box_size = self.GroupBoxList[i][j].size()
                    # 调整 pixmap 的大小以适应组框的大小，并保持宽高比
                    scaled_pixmap = self.ImgLabelList[i][j].original_pixmap.scaled(group_box_size,
                                                                                   Qt.AspectRatioMode.KeepAspectRatio,
                                                                                   Qt.TransformationMode.SmoothTransformation)
                    # 设置调整大小的 pixmap 到标签
                    self.ImgLabelList[i][j].setPixmap(scaled_pixmap)

    def pil_image_to_pixmap(self, pil_img):
        # 将 PIL 图像转换为 QByteArray
        if pil_img.mode != "RGBA":
            pil_img = pil_img.convert("RGBA")
        data = pil_img.tobytes("raw", "BGRA")
        qim = QImage(data, pil_img.width, pil_img.height, QImage.Format.Format_ARGB32)
        pixmap = QPixmap.fromImage(qim)
        return pixmap


class CustomDelegate(QStyledItemDelegate):
    def sizeHint(self, option, index):
        size = super().sizeHint(option, index)
        size.setHeight(size.height() * 2)  # 设置行高为原来的两倍
        return size


class BlurWindow(QMainWindow):
    def __init__(self, window_list):
        super().__init__()

        self.setGeometry(100, 100, 600, 400)
        self.setStyleSheet("background-color: black")

        # 创建侧边栏和堆叠小部件
        self.sidebar = QListWidget()
        self.sidebar.setItemDelegate(CustomDelegate(self))
        self.sidebar.setMaximumWidth(150)
        self.sidebar.setStyleSheet("""
            QListWidget {
                color: white;
                background-color: black;
                font-size: 15px;
            }
            QListWidget::item:selected {
                background-color: blue;
            }
        """)
        self.stacked_widget = QStackedWidget()

        # 创建图像处理界面
        for i in range(len(window_list)):
            image_processing_widget = ImageProcessingWidget(1)
            self.stacked_widget.addWidget(image_processing_widget)

        self.sidebar.addItems(window_list)
        self.sidebar.currentRowChanged.connect(self.display_widget)
        self.sidebar.setStyleSheet("color: white; font-size: 15px")
        # 主布局
        main_layout = QHBoxLayout()
        main_layout.addWidget(self.sidebar)
        main_layout.addWidget(self.stacked_widget, 1)

        # 设置中心小部件
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def display_widget(self, index):
        self.stacked_widget.setCurrentIndex(index)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window_list = ["界面1", "界面2", "界面3"]
    window = BlurWindow(window_list)
    window.resize(1200, 800)
    window.setWindowTitle("demo")
    window.show()
    sys.exit(app.exec())
