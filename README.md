# Mini-GPT 智能对话项目

一个基于 PyTorch 实现的轻量级 GPT 语言模型。项目包含了完整的 Transformer 核心架构实现、中文语料预处理、模型预训练以及下游微调（Finetune）的完整流水线。

---

## 🚀 功能特性

- **纯手工打造 Transformer**：基于 PyTorch 实现了多头自注意力机制（Multi-Head Attention）及前馈神经网络。
- **丰富的中文语料**：内置了包括情感、食物、文学、心理、幽默等多领域的中文对话语料（`.yml` 格式）。
- **两阶段训练**：支持从零开始的预训练（Pre-train）以及结合特定领域的微调（Fine-tune）。

---

## 🛠️ 环境准备

在运行项目之前，请确保你的机器上已安装 Python 3.8+ 以及 PyTorch。

1. **克隆项目到本地**：
```bash
git clone https://github.com/Tjy0311/my-Minigpt-projrct.git
cd my-Minigpt-projrct  
2.安装依赖环境（建议在虚拟环境中运行）：S
```bash
pip install torch pyyaml
📂 项目结构
├── chinese/               # 中文多领域对话语料库 (.yml)
├── convert.py             # 数据预处理与格式转换脚本
├── train.py               # 预训练主程序
├── finetune.py            # 模型微调主程序
├── transformer.py         # Transformer 模型核心架构实现
├── input.txt              # 基础训练文本输入
├── pretrained_gpt.pth     # 预训练模型权重（约 69MB）
├── finetuned_gpt.pth      # 微调后的模型权重（约 69MB）
└── README.md              # 项目说明文档
🏋️‍♂️ 如何运行与复现
第一步：数据准备
运行 convert.py 将 chinese/ 文件夹下的 YAML 语料或 input.txt 转换为模型可读取的训练数据：
```bash
python convert.py
第二步：模型预训练
运行 train.py 开始从零训练基础语言模型。训练完成后，会在本地生成 pretrained_gpt.pth 权重文件：
Bash
    python train.py
第三步：下游任务微调
如果你想让模型在特定的中文对话上表现更好，可以运行 finetune.py 进行微调，生成 finetuned_gpt.pth：
Bash
    python finetune.py
🤝 贡献与交流
欢迎提交 Issue 或 Pull Request 来一起完善这个轻量级的 GPT 实现！如果你觉得这个项目对你有帮助，不妨点个 ⭐ Star 吧！
