import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformer import MiniGPT, GPTConfig


class TextDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.block_size]
        y = self.data[idx + 1: idx + self.block_size + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def main():
    print("=" * 60)
    print("🚀 (Pre-training)")
    print("=" * 60)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with open('input.txt', 'r', encoding='utf-8') as f:
        raw_text = f.read()

    chars = sorted(list(set(raw_text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    data = [stoi[c] for c in raw_text]

    print(f"📊 语料总字数: {len(data)} | 去重词汇表大小(Vocab): {vocab_size}")

    config = GPTConfig()
    config.vocab_size = vocab_size
    model = MiniGPT(config).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=5e-4)
    scaler = GradScaler()

    dataset = TextDataset(data, config.block_size)
    batch_size = 32
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    epochs = 40

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
            print(f"🔄 Epoch [{epoch + 1}/{epochs}] | 预训练 Loss: {total_loss / len(dataloader):.4f}")

    torch.save({
        'model_state_dict': model.state_dict(),
        'stoi': stoi,
        'itos': itos
    }, 'pretrained_gpt.pth')
    print("✅ 预训练权重已成功保存至 pretrained_gpt.pth！\n")


if __name__ == "__main__":
    main()