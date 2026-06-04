import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformer import MiniGPT, GPTConfig
import os


class SFTDataset(Dataset):
    def __init__(self, qa_pairs, stoi, block_size):
        self.data = []
        unk_id = stoi.get('<UNK>', 0)

        for q, a in qa_pairs:
            text = f"问:{q} 答:{a}。"
            tokens = [stoi.get(c, unk_id) for c in text]
            if len(tokens) > block_size + 1:
                tokens = tokens[:block_size + 1]
            else:
                tokens += [0] * (block_size + 1 - len(tokens))
            self.data.append(tokens)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data[idx]
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        return x, y


def extract_qa_from_txt(filepath):
    qa_pairs = []
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if line.startswith('问:') and '答:' in line:
            parts = line.split('答:')
            if len(parts) == 2:
                q = parts[0].replace('问:', '').strip()
                a = parts[1].strip()
                if q and a:
                    qa_pairs.append((q, a))
    return qa_pairs


def main():
    print("=" * 70)
    print("🎯 [2/2] 启动监督微调 (SFT) - 注入问答格式与高情商逻辑")
    print("=" * 70)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not os.path.exists('pretrained_gpt.pth'):
        print("❌ 找不到 pretrained_gpt.pth！")
        return

    checkpoint = torch.load('pretrained_gpt.pth', map_location=device)
    stoi = checkpoint['stoi']
    itos = checkpoint['itos']
    vocab_size = len(stoi)

    qa_pairs = extract_qa_from_txt('input.txt')
    print(f"🎯 成功提取 {len(qa_pairs)} 条高质量监督问答对。")

    config = GPTConfig()
    config.vocab_size = vocab_size
    model = MiniGPT(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # 保护性低学习率 1e-4
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scaler = GradScaler()

    dataset = SFTDataset(qa_pairs, stoi, config.block_size)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model.train()
    epochs = 80

    for epoch in range(epochs):
        total_loss = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with autocast():
                logits, loss = model(x, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"🎯 SFT Epoch [{epoch + 1}/{epochs}] | 对齐 Loss: {total_loss / len(dataloader):.4f}")

    torch.save({
        'model_state_dict': model.state_dict(),
        'stoi': stoi,
        'itos': itos
    }, 'finetuned_gpt.pth')
    print("✅ 终极微调完成！模型已保存在 finetuned_gpt.pth！\n")


if __name__ == "__main__":
    main()