import sys
import threading
import torch
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from transformer import MiniGPT, GPTConfig
import os


class CloudWindow(QWidget):
    update_chat_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("MiniGPT 梦幻云端")
        self.resize(700, 850)
        self.setStyleSheet("""
            CloudWindow { background: qradialgradient(cx:0.5, cy:0.5, radius:0.8, fx:0.5, fy:0.5, stop:0 #FFD1DC, stop:1 #B5EAD7); }
            QTextEdit { background: rgba(255, 255, 255, 0.4); border-radius: 25px; padding: 20px; font-size: 18px; border: none; }
            QLineEdit { background: rgba(255, 255, 255, 0.8); border-radius: 25px; padding: 15px; font-size: 18px; }
            QPushButton { background: #FF9AA2; border-radius: 25px; color: white; font-weight: bold; font-size: 18px; }
        """)

        # 布局
        layout = QVBoxLayout(self)
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        layout.addWidget(self.chat_display)

        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("请输入内容...")
        layout.addWidget(self.input_field)

        self.send_btn = QPushButton("发 送")
        self.send_btn.setFixedHeight(60)
        self.send_btn.clicked.connect(self.send_message)
        layout.addWidget(self.send_btn)

        self.update_chat_signal.connect(self.update_chat)
        self.init_model()

    def create_bot_icon(self):
        """代码生成一个简约机器人图标，无需外部图片"""
        pixmap = QPixmap(50, 50)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QColor("#4A90E2"))
        painter.drawRoundedRect(5, 5, 40, 40, 10, 10)  # 头部
        painter.setBrush(Qt.white)
        painter.drawEllipse(12, 12, 10, 10)  # 左眼
        painter.drawEllipse(28, 12, 10, 10)  # 右眼
        painter.end()
        return pixmap

    def update_chat(self, text):
        # 使用文档对象插入图标
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.chat_display.setTextCursor(cursor)

        # 插入机器人图标
        self.chat_display.document().addResource(QTextDocument.ImageResource, QUrl("bot"), self.create_bot_icon())
        self.chat_display.textCursor().insertHtml('<img src="bot" width="30" height="30"> ')
        self.chat_display.append(f"<b>MiniGPT:</b> {text}<br><br>")

    def init_model(self):
        # 模拟模型加载逻辑
        threading.Thread(target=self._load_model, daemon=True).start()

    def _load_model(self):
        # 这里放置你的模型加载逻辑
        self.update_chat_signal.emit("引擎已就绪，我是你的专属云端助手！")

    def send_message(self):
        text = self.input_field.text()
        if not text: return
        self.chat_display.append(f"<div align='right'><b>我:</b> {text}</div>")
        self.input_field.clear()
        # 这里触发推理逻辑...
        self.update_chat_signal.emit("这是一条来自云端的温柔回复。")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CloudWindow()
    window.show()
    sys.exit(app.exec_())
