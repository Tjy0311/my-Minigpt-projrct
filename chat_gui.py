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

        # 样式表：大幅放大字体 (font-size: 18px+)
        self.setStyleSheet("""
            CloudWindow {
                background: qradialgradient(cx:0.5, cy:0.5, radius:0.8, fx:0.5, fy:0.5, 
                            stop:0 #FFD1DC, stop:0.3 #B5EAD7, stop:0.6 #C7CEEA, stop:1 #FFDAC1);
            }
            QTextEdit {
                background: rgba(255, 255, 255, 0.5);
                border: 2px solid rgba(255, 255, 255, 0.7);
                border-radius: 30px;
                padding: 20px;
                color: #2C3E50;
                font-family: 'Microsoft YaHei';
                font-size: 18px; /* 字体放大 */
            }
            QLineEdit {
                background: rgba(255, 255, 255, 0.8);
                border: none;
                border-radius: 30px;
                padding: 20px 30px;
                color: #2C3E50;
                font-size: 18px; /* 字体放大 */
            }
            QPushButton {
                background: #FF9AA2;
                border-radius: 30px;
                color: white;
                font-weight: bold;
                font-size: 20px; /* 按钮字体放大 */
            }
        """)

        self.update_chat_signal.connect(self.update_chat)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.load_model_thread()

        # 布局
        layout = QVBoxLayout()
        layout.setContentsMargins(50, 50, 50, 50)
        layout.setSpacing(25)

        # 标题区域
        title = QLabel("MiniGPT")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: white; font-size: 28px; font-weight: bold; background: none;")
        layout.addWidget(title)

        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        layout.addWidget(self.chat_display)

        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("请输入问题...")
        self.input_field.returnPressed.connect(self.send_message)
        layout.addWidget(self.input_field)

        self.send_btn = QPushButton("发 送")
        self.send_btn.setMinimumHeight(65)
        self.send_btn.clicked.connect(self.send_message)
        layout.addWidget(self.send_btn)

        self.setLayout(layout)

    def update_chat(self, text):
        # 检查是否有机器人图片
        icon_path = "bot_icon.png"
        if os.path.exists(icon_path):
            # 插入图片：把机器人图标放在回复前
            cursor = self.chat_display.textCursor()
            self.chat_display.moveCursor(QTextCursor.End)
            self.chat_display.textCursor().insertHtml(f'<img src="{icon_path}" width="30" height="30"> ')

        # 加上机器人名称，并放大字体显示
        self.chat_display.append(f"<b>🤖 MiniGPT:</b> {text}")

    # ... 其他 load_model_thread 和 run_inference 保持不变 ...
    def load_model_thread(self):
        def _load():
            if not os.path.exists('finetuned_gpt.pth'):
                self.update_chat_signal.emit("未找到权重文件！")
                return
            checkpoint = torch.load('finetuned_gpt.pth', map_location=self.device)
            self.stoi, self.itos = checkpoint['stoi'], checkpoint['itos']
            config = GPTConfig()
            config.vocab_size = len(self.stoi)
            self.model = MiniGPT(config).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            self.update_chat_signal.emit("引擎已就绪，随时待命。")

        threading.Thread(target=_load, daemon=True).start()

    def send_message(self):
        user_text = self.input_field.text()
        if not user_text: return
        self.chat_display.append(f"<div align='right'><b>我:</b> {user_text}</div>")
        self.input_field.clear()
        threading.Thread(target=self.run_inference, args=(user_text,), daemon=True).start()

    def run_inference(self, prompt):
        input_ids = [self.stoi.get(c, 0) for c in f"问:{prompt} 答:"]
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        ids = self.model.generate(input_tensor, max_new_tokens=150, temperature=0.1)
        text = "".join([self.itos.get(int(i), "") for i in ids[0]])
        answer = text.split("答:")[1] if "答:" in text else text
        self.update_chat_signal.emit(answer.split("。")[0] + "。")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CloudWindow()
    window.show()
    sys.exit(app.exec_())