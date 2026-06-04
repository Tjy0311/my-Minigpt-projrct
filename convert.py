import os
import sys
import yaml  # 如果提示缺少该库，请在 PyCharm 终端执行: pip install pyyaml

print("=" * 60)
print("       🤖 MiniGPT 数据工程：ChatterBot 语料智能清洗流 🤖       ")
print("=" * 60)
print("[系统日志] 正在启动智能路径雷达，全盘检索简中对话数据集...")

# ==========================================
# 1. 智能路径雷达（线性搜索）
# ==========================================
corpus_dir = None
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()

# 第一轮：深度搜寻当前项目目录下的 chinese 文件夹
for root, dirs, files in os.walk(current_dir):
    if os.path.basename(root).lower() == 'chinese':
        if any(f.endswith('.yml') for f in files):
            corpus_dir = root
            break

# 第二轮：扩大范围搜寻
if not corpus_dir:
    for root, dirs, files in os.walk('.'):
        if os.path.basename(root).lower() == 'chinese':
            if any(f.endswith('.yml') for f in files):
                corpus_dir = os.path.abspath(root)
                break

# 第三轮：绝对路径硬核保底机制
if not corpus_dir:
    print("\n[⚠️ 提示] 雷达未在当前项目内搜寻到 'chinese' 文件夹。")
    print(" -> 正在尝试触发硬编码物理绝对路径保底...")
    corpus_dir = r"D:\PycharmProjects\pythonProject7\chatterbot-corpus-master\chatterbot_corpus\data\chinese"

output_txt = "chatterbot_daily.txt"

# ==========================================
# 2. 核心数据清洗与对齐流
# ==========================================
if not os.path.exists(corpus_dir):
    print(f"\n❌ [编译中断]：未能成功定位语料库！请确认 chinese 文件夹是否已解压。")
    print(f"当前尝试检索的保底路径为: {corpus_dir}")
else:
    print(f"✅ [锁定成功] 成功定位到简中原始语料目录：\n👉 {corpus_dir}\n")
    print("[系统日志] 开始执行无损语法清洗与前缀注意力控制符（问:答:）对齐...")

    file_count = 0
    total_pairs = 0

    with open(output_txt, "w", encoding="utf-8") as out_f:
        # 顺畅遍历锁定的文件夹
        for file_name in os.listdir(corpus_dir):
            if file_name.endswith(".yml"):
                file_path = os.path.join(corpus_dir, file_name)

                with open(file_path, "r", encoding="utf-8") as f:
                    try:
                        data = yaml.safe_load(f)
                        if data and "conversations" in data:
                            for conv in data["conversations"]:
                                if len(conv) >= 2:
                                    # 数据清洗核心：剔除换行符、首尾空白，确保单行高内聚
                                    question = str(conv[0]).replace('\n', '').strip()
                                    answer = str(conv[1]).replace('\n', '').strip()

                                    # 按照大模型最熟悉的 SFT Token 边界进行序列化写入
                                    out_f.write(f"问:{question} 答:{answer}。\n")
                                    total_pairs += 1
                            file_count += 1
                    except Exception as e:
                        print(f"⚠️ [跳过兼容] 文件 {file_name} 解析异常: {e}")

    print(f"\n🎉 【数据工程圆满成功】")
    print(f"📊 成功无损清洗了 {file_count} 个全品类学科领域的 YAML 语料文件")
    print(f"📝 共计提纯出 {total_pairs} 条纯净口语化日常高质量对话条件概率链条")
    print(f"📁 纯净版日常对话 TXT 已安全保存至: {os.path.abspath(output_txt)}")
    print("=" * 60)