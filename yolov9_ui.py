import os
import sys
import cv2
import numpy as np
import torch
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QPushButton, 
                             QVBoxLayout, QHBoxLayout, QLabel, QFileDialog, 
                             QComboBox, QSlider, QCheckBox, QMessageBox, QTabWidget,
                             QGridLayout, QGroupBox, QSpinBox, QDoubleSpinBox,
                             QSplitter, QFrame, QProgressBar, QStyle)
from PyQt5.QtGui import QPixmap, QImage, QPalette, QColor
from PyQt5.QtCore import Qt, QTimer, pyqtSlot, QThread, pyqtSignal, QSize
import time
from pathlib import Path

# 导入YOLOv9需要的模块
from models.common import DetectMultiBackend
from utils.general import (check_img_size, non_max_suppression, scale_boxes)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device
from utils.augmentations import letterbox

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    fps_signal = pyqtSignal(float)
    
    def __init__(self, source='0', model=None, imgsz=(640, 640), conf_thres=0.25, 
                 iou_thres=0.45, classes=None, hide_conf=False, hide_labels=False):
        super().__init__()
        self.source = source
        self.model = model
        self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.classes = classes
        self.hide_conf = hide_conf
        self.hide_labels = hide_labels
        self.running = False
        self.vid_stride = 1
        self.frame_count = 0
    
    def run(self):
        if self.source.isdigit():
            self.source = int(self.source)
        
        cap = cv2.VideoCapture(self.source)
        
        if not cap.isOpened():
            print(f"Error: Could not open video source {self.source}")
            return
            
        # 设置帧率计算
        fps_start_time = time.time()
        fps_frame_count = 0
        fps = 0
        
        self.running = True
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
                
            # 处理帧计数
            self.frame_count += 1
            if self.frame_count % self.vid_stride != 0:
                continue
                
            # 准备图像用于检测
            img = letterbox(frame, self.imgsz, stride=self.model.stride)[0]
            img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(self.model.device)
            img = img.float()  # uint8 to fp16/32
            img /= 255  # 0 - 255 to 0.0 - 1.0
            if len(img.shape) == 3:
                img = img[None]  # expand for batch dim
                
            # 进行推理
            pred = self.model(img, augment=False, visualize=False)
            
            # NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes)
            
            # 处理检测结果
            for i, det in enumerate(pred):  # per image
                annotator = Annotator(frame, line_width=3, example=str(self.model.names))
                if len(det):
                    # 将框从img_size重新缩放到im0大小
                    det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()
                    
                    # 在图像上绘制检测结果
                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)  # 整数类别
                        label = None if self.hide_labels else (self.model.names[c] if self.hide_conf else f'{self.model.names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
            
            # 计算FPS
            fps_frame_count += 1
            if fps_frame_count >= 10:  # 每10帧更新一次FPS
                fps = fps_frame_count / (time.time() - fps_start_time)
                fps_start_time = time.time()
                fps_frame_count = 0
                self.fps_signal.emit(fps)
                
            # 发送处理后的帧
            self.change_pixmap_signal.emit(annotator.result())
            
        cap.release()
    
    def stop(self):
        self.running = False
        self.wait()

class YOLOv9UI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLOv9 Detection UI")
        self.setMinimumSize(1200, 800)
        
        # 模型设置
        self.device = select_device('0')
        self.model = None
        self.imgsz = (640, 640)
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.classes = None
        self.hide_labels = False
        self.hide_conf = False
        
        # 视频处理
        self.video_thread = None
        
        # 创建UI
        self.init_ui()
        
        # 加载默认模型
        self.weights_path = os.path.join('weights', 'yolov9-t-converted.pt')
        if os.path.exists(self.weights_path):
            self.model_combo.setCurrentText("YOLOv9-T")
            self.load_model()
        
    def init_ui(self):
        # 创建主布局
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        self.setCentralWidget(main_widget)
        
        # 创建标签页
        tab_widget = QTabWidget()
        
        # 创建各个功能标签页
        image_tab = self.create_image_tab()
        video_tab = self.create_video_tab()
        camera_tab = self.create_camera_tab()
        
        # 添加标签页
        tab_widget.addTab(image_tab, "图像检测")
        tab_widget.addTab(video_tab, "视频检测")
        tab_widget.addTab(camera_tab, "摄像头检测")
        
        # 创建通用设置面板
        settings_group = self.create_settings_panel()
        
        # 添加到主布局
        main_layout.addWidget(tab_widget, 3)
        main_layout.addWidget(settings_group, 1)
        
    def create_image_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 上部显示区域
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setStyleSheet("background-color: #222; color: white;")
        self.image_label.setText("在这里显示图像检测结果")
        
        # 按钮区域
        button_layout = QHBoxLayout()
        
        self.load_image_btn = QPushButton("加载图像")
        self.load_image_btn.clicked.connect(self.load_image)
        
        self.detect_image_btn = QPushButton("检测")
        self.detect_image_btn.clicked.connect(self.detect_image)
        self.detect_image_btn.setEnabled(False)
        
        self.save_image_btn = QPushButton("保存结果")
        self.save_image_btn.clicked.connect(self.save_image_result)
        self.save_image_btn.setEnabled(False)
        
        button_layout.addWidget(self.load_image_btn)
        button_layout.addWidget(self.detect_image_btn)
        button_layout.addWidget(self.save_image_btn)
        
        # 状态区域
        self.image_status_label = QLabel("准备就绪")
        
        layout.addWidget(self.image_label, 4)
        layout.addLayout(button_layout)
        layout.addWidget(self.image_status_label)
        
        return tab
    
    def create_video_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 上部显示区域
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("background-color: #222; color: white;")
        self.video_label.setText("在这里显示视频检测结果")
        
        # 按钮区域
        button_layout = QHBoxLayout()
        
        self.load_video_btn = QPushButton("加载视频")
        self.load_video_btn.clicked.connect(self.load_video)
        
        self.start_video_btn = QPushButton("开始检测")
        self.start_video_btn.clicked.connect(self.start_video_detection)
        self.start_video_btn.setEnabled(False)
        
        self.stop_video_btn = QPushButton("停止")
        self.stop_video_btn.clicked.connect(self.stop_video_detection)
        self.stop_video_btn.setEnabled(False)
        
        button_layout.addWidget(self.load_video_btn)
        button_layout.addWidget(self.start_video_btn)
        button_layout.addWidget(self.stop_video_btn)
        
        # FPS显示
        self.video_fps_label = QLabel("FPS: 0")
        
        layout.addWidget(self.video_label, 4)
        layout.addLayout(button_layout)
        layout.addWidget(self.video_fps_label)
        
        return tab
    
    def create_camera_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 上部显示区域
        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setStyleSheet("background-color: #222; color: white;")
        self.camera_label.setText("在这里显示摄像头检测结果")
        
        # 按钮和选择区域
        control_layout = QHBoxLayout()
        
        self.camera_combo = QComboBox()
        self.camera_combo.addItem("摄像头 0")
        self.camera_combo.addItem("摄像头 1")
        self.camera_combo.addItem("摄像头 2")
        
        self.start_camera_btn = QPushButton("开始检测")
        self.start_camera_btn.clicked.connect(self.start_camera_detection)
        
        self.stop_camera_btn = QPushButton("停止")
        self.stop_camera_btn.clicked.connect(self.stop_camera_detection)
        self.stop_camera_btn.setEnabled(False)
        
        control_layout.addWidget(self.camera_combo)
        control_layout.addWidget(self.start_camera_btn)
        control_layout.addWidget(self.stop_camera_btn)
        
        # FPS显示
        self.camera_fps_label = QLabel("FPS: 0")
        
        layout.addWidget(self.camera_label, 4)
        layout.addLayout(control_layout)
        layout.addWidget(self.camera_fps_label)
        
        return tab
    
    def create_settings_panel(self):
        group_box = QGroupBox("模型设置")
        layout = QGridLayout(group_box)
        
        # 模型选择
        layout.addWidget(QLabel("模型:"), 0, 0)
        self.model_combo = QComboBox()
        self.model_combo.addItems(["YOLOv9-T", "YOLOv9-S", "YOLOv9-M", "YOLOv9-C", "YOLOv9-E"])
        self.model_combo.currentIndexChanged.connect(self.on_model_changed)
        layout.addWidget(self.model_combo, 0, 1)
        
        # 加载自定义模型
        self.custom_model_btn = QPushButton("加载自定义模型")
        self.custom_model_btn.clicked.connect(self.load_custom_model)
        layout.addWidget(self.custom_model_btn, 0, 2)
        
        # 置信度阈值
        layout.addWidget(QLabel("置信度阈值:"), 1, 0)
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(1, 99)
        self.conf_slider.setValue(int(self.conf_thres * 100))
        self.conf_slider.valueChanged.connect(self.update_conf_thres)
        layout.addWidget(self.conf_slider, 1, 1)
        self.conf_value_label = QLabel(f"{self.conf_thres:.2f}")
        layout.addWidget(self.conf_value_label, 1, 2)
        
        # IoU阈值
        layout.addWidget(QLabel("IoU阈值:"), 2, 0)
        self.iou_slider = QSlider(Qt.Horizontal)
        self.iou_slider.setRange(1, 99)
        self.iou_slider.setValue(int(self.iou_thres * 100))
        self.iou_slider.valueChanged.connect(self.update_iou_thres)
        layout.addWidget(self.iou_slider, 2, 1)
        self.iou_value_label = QLabel(f"{self.iou_thres:.2f}")
        layout.addWidget(self.iou_value_label, 2, 2)
        
        # 图像大小
        layout.addWidget(QLabel("图像大小:"), 3, 0)
        self.imgsz_combo = QComboBox()
        self.imgsz_combo.addItems(["320", "416", "512", "640", "1280"])
        self.imgsz_combo.setCurrentText(str(self.imgsz[0]))
        self.imgsz_combo.currentTextChanged.connect(self.update_imgsz)
        layout.addWidget(self.imgsz_combo, 3, 1)
        
        # 显示选项
        self.hide_labels_check = QCheckBox("隐藏标签")
        self.hide_labels_check.stateChanged.connect(self.update_display_options)
        layout.addWidget(self.hide_labels_check, 4, 0)
        
        self.hide_conf_check = QCheckBox("隐藏置信度")
        self.hide_conf_check.stateChanged.connect(self.update_display_options)
        layout.addWidget(self.hide_conf_check, 4, 1)
        
        return group_box
    
    def update_conf_thres(self, value):
        self.conf_thres = value / 100
        self.conf_value_label.setText(f"{self.conf_thres:.2f}")
    
    def update_iou_thres(self, value):
        self.iou_thres = value / 100
        self.iou_value_label.setText(f"{self.iou_thres:.2f}")
    
    def update_imgsz(self, value):
        size = int(value)
        self.imgsz = (size, size)
    
    def update_display_options(self):
        self.hide_labels = self.hide_labels_check.isChecked()
        self.hide_conf = self.hide_conf_check.isChecked()
    
    def on_model_changed(self, index):
        model_name = self.model_combo.currentText()
        model_map = {
            "YOLOv9-T": "yolov9-t-converted.pt",
            "YOLOv9-S": "yolov9-s-converted.pt",
            "YOLOv9-M": "yolov9-m-converted.pt",
            "YOLOv9-C": "yolov9-c-converted.pt",
            "YOLOv9-E": "yolov9-e-converted.pt"
        }
        
        if model_name in model_map:
            self.weights_path = os.path.join('weights', model_map[model_name])
            if os.path.exists(self.weights_path):
                self.load_model()
            else:
                QMessageBox.warning(self, "模型文件不存在", 
                                   f"模型文件 {self.weights_path} 不存在，请下载后放入weights文件夹")
    
    def load_custom_model(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件", "", "PyTorch Model (*.pt *.pth);;All Files (*)", options=options
        )
        
        if file_name:
            self.weights_path = file_name
            self.load_model()
    
    def load_model(self):
        try:
            # 如果正在运行视频或摄像头检测，先停止
            if self.video_thread and self.video_thread.running:
                self.stop_video_detection()
            
            QApplication.setOverrideCursor(Qt.WaitCursor)
            self.model = DetectMultiBackend(self.weights_path, device=self.device)
            self.imgsz = check_img_size(self.imgsz, s=self.model.stride)
            
            # 更新UI状态
            self.image_status_label.setText(f"模型已加载: {os.path.basename(self.weights_path)}")
            self.detect_image_btn.setEnabled(True)
            QApplication.restoreOverrideCursor()
            
            QMessageBox.information(self, "模型加载成功", 
                                   f"成功加载模型: {os.path.basename(self.weights_path)}")
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "模型加载失败", f"错误: {str(e)}")
    
    def load_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self, "选择图像", "", "Images (*.png *.jpg *.jpeg *.bmp);;All Files (*)", options=options
        )
        
        if file_name:
            self.image_path = file_name
            # 显示原始图像
            pixmap = QPixmap(file_name)
            self.image_label.setPixmap(pixmap.scaled(
                self.image_label.width(), self.image_label.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))
            self.detect_image_btn.setEnabled(True)
            self.image_status_label.setText(f"图像已加载: {os.path.basename(file_name)}")
    
    def detect_image(self):
        if not hasattr(self, 'image_path') or not self.model:
            return
        
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            
            # 读取图像
            img0 = cv2.imread(self.image_path)
            assert img0 is not None, f"Image not found: {self.image_path}"
            
            # 准备图像用于检测
            img = letterbox(img0, self.imgsz, stride=self.model.stride)[0]
            img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(self.device)
            img = img.float()  # uint8 to fp16/32
            img /= 255  # 0 - 255 to 0.0 - 1.0
            if len(img.shape) == 3:
                img = img[None]  # expand for batch dim
                
            # 进行推理
            start_time = time.time()
            pred = self.model(img, augment=False, visualize=False)
            
            # NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes)
            
            # 处理检测结果
            det_img = img0.copy()
            for i, det in enumerate(pred):  # per image
                annotator = Annotator(det_img, line_width=3, example=str(self.model.names))
                if len(det):
                    # 将框从img_size重新缩放到im0大小
                    det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], det_img.shape).round()
                    
                    # 在图像上绘制检测结果
                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)  # 整数类别
                        label = None if self.hide_labels else (self.model.names[c] if self.hide_conf else f'{self.model.names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
            
            inference_time = time.time() - start_time
            
            # 将结果转换为QPixmap显示
            self.result_img = annotator.result()
            h, w, ch = self.result_img.shape
            bytes_per_line = ch * w
            qt_img = QImage(self.result_img.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(qt_img)
            self.image_label.setPixmap(pixmap.scaled(
                self.image_label.width(), self.image_label.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))
            
            # 更新UI状态
            object_count = sum(len(d) for d in pred)
            self.image_status_label.setText(f"检测完成: 找到 {object_count} 个对象，耗时 {inference_time:.3f} 秒")
            self.save_image_btn.setEnabled(True)
            
            QApplication.restoreOverrideCursor()
            
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "检测失败", f"错误: {str(e)}")
    
    def save_image_result(self):
        if not hasattr(self, 'result_img'):
            return
        
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(
            self, "保存结果图像", "", "Images (*.png *.jpg *.jpeg);;All Files (*)", options=options
        )
        
        if file_name:
            cv2.imwrite(file_name, self.result_img)
            QMessageBox.information(self, "保存成功", f"结果已保存至: {file_name}")
    
    def load_video(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self, "选择视频", "", "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)", options=options
        )
        
        if file_name:
            self.video_path = file_name
            self.start_video_btn.setEnabled(True)
            self.video_fps_label.setText(f"视频已加载: {os.path.basename(file_name)}")
    
    def start_video_detection(self):
        if not hasattr(self, 'video_path') or not self.model:
            return
        
        # 创建并启动视频处理线程
        self.video_thread = VideoThread(
            source=self.video_path,
            model=self.model,
            imgsz=self.imgsz,
            conf_thres=self.conf_thres,
            iou_thres=self.iou_thres,
            classes=self.classes,
            hide_conf=self.hide_conf,
            hide_labels=self.hide_labels
        )
        self.video_thread.change_pixmap_signal.connect(self.update_video_frame)
        self.video_thread.fps_signal.connect(self.update_video_fps)
        self.video_thread.start()
        
        # 更新UI状态
        self.start_video_btn.setEnabled(False)
        self.stop_video_btn.setEnabled(True)
        self.load_video_btn.setEnabled(False)
    
    def stop_video_detection(self):
        if self.video_thread and self.video_thread.running:
            self.video_thread.stop()
            
        # 更新UI状态
        self.start_video_btn.setEnabled(True)
        self.stop_video_btn.setEnabled(False)
        self.load_video_btn.setEnabled(True)
    
    def start_camera_detection(self):
        if not self.model:
            QMessageBox.warning(self, "模型未加载", "请先加载模型")
            return
        
        camera_index = self.camera_combo.currentIndex()
        
        # 创建并启动摄像头处理线程
        self.video_thread = VideoThread(
            source=str(camera_index),
            model=self.model,
            imgsz=self.imgsz,
            conf_thres=self.conf_thres,
            iou_thres=self.iou_thres,
            classes=self.classes,
            hide_conf=self.hide_conf,
            hide_labels=self.hide_labels
        )
        self.video_thread.change_pixmap_signal.connect(self.update_camera_frame)
        self.video_thread.fps_signal.connect(self.update_camera_fps)
        self.video_thread.start()
        
        # 更新UI状态
        self.start_camera_btn.setEnabled(False)
        self.stop_camera_btn.setEnabled(True)
        self.camera_combo.setEnabled(False)
    
    def stop_camera_detection(self):
        if self.video_thread and self.video_thread.running:
            self.video_thread.stop()
            
        # 更新UI状态
        self.start_camera_btn.setEnabled(True)
        self.stop_camera_btn.setEnabled(False)
        self.camera_combo.setEnabled(True)
    
    @pyqtSlot(np.ndarray)
    def update_video_frame(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.video_label.setPixmap(qt_img)
    
    @pyqtSlot(float)
    def update_video_fps(self, fps):
        self.video_fps_label.setText(f"FPS: {fps:.1f}")
    
    @pyqtSlot(np.ndarray)
    def update_camera_frame(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.camera_label.setPixmap(qt_img)
    
    @pyqtSlot(float)
    def update_camera_fps(self, fps):
        self.camera_fps_label.setText(f"FPS: {fps:.1f}")
    
    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_qt_format.scaled(self.video_label.width(), self.video_label.height(), 
                                        Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)
    
    def closeEvent(self, event):
        # 停止所有运行中的线程
        if self.video_thread and self.video_thread.running:
            self.video_thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = YOLOv9UI()
    window.show()
    sys.exit(app.exec_()) 