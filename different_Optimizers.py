# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import datasets, transforms, models
# from torch.utils.data import DataLoader
# import matplotlib.pyplot as plt
# import time
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Using device: {device}")
#
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
#
# train_dataset = datasets.STL10(root='./data', split='train', download=True, transform=transform)
# test_dataset = datasets.STL10(root='./data', split='test', download=True, transform=transform)
#
# batch_size = 32
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
#
# def get_model(num_classes=10):
#     model = models.resnet18(pretrained=False)  # 不用预训练权重以保证公平
#     model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)  # 适配小图像
#     model.maxpool = nn.Identity()  # 移除 maxpool 以防特征过小
#     model.fc = nn.Linear(model.fc.in_features, num_classes)
#     return model.to(device)
#
#
# optimizers_config = {
#     "SGD": lambda params: optim.SGD(params, lr=0.01, momentum=0.9),
#     "Adam": lambda params: optim.Adam(params, lr=0.001),
#     "RMSprop": lambda params: optim.RMSprop(params, lr=0.001, alpha=0.99),
#     "Adagrad": lambda params: optim.Adagrad(params, lr=0.01),
#     "AdamW": lambda params: optim.AdamW(params, lr=0.001, weight_decay=0.01),
# }
#
# criterion = nn.CrossEntropyLoss()
#
# def train_model(model, optimizer, train_loader, epochs=20):
#     model.train()
#     losses = []
#     for epoch in range(epochs):
#         total_loss = 0
#         for images, labels in train_loader:
#             images, labels = images.to(device), labels.to(device)
#             optimizer.zero_grad()
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#         avg_loss = total_loss / len(train_loader)
#         losses.append(avg_loss)
#         print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
#     return losses
#
# def evaluate_model(model, test_loader):
#     model.eval()
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for images, labels in test_loader:
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             _, predicted = torch.max(outputs, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#     accuracy = 100 * correct / total
#     print(f"Test Accuracy: {accuracy:.2f}%")
#     return accuracy
#
#
# results = {}
#
# for opt_name, opt_fn in optimizers_config.items():
#     print(f"\n{'=' * 50}")
#     print(f"Training with {opt_name}")
#     print('=' * 50)
#
#     # 重新初始化模型（确保公平）
#     model = get_model()
#     optimizer = opt_fn(model.parameters())
#
#     # 训练
#     start_time = time.time()
#     train_losses = train_model(model, optimizer, train_loader, epochs=20)
#     train_time = time.time() - start_time
#
#     # 测试
#     test_acc = evaluate_model(model, test_loader)
#
#     # 保存结果
#     results[opt_name] = {
#         'train_losses': train_losses,
#         'test_accuracy': test_acc,
#         'train_time': train_time
#     }
#
# 绘制训练损失曲线
plt.figure(figsize=(12, 5))

# 子图1：损失曲线
plt.subplot(1, 2, 1)
for opt_name, res in results.items():
    plt.plot(res['train_losses'], label=f"{opt_name} ({res['test_accuracy']:.1f}%)")
plt.title("Training Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

# 子图2：最终准确率柱状图
plt.subplot(1, 2, 2)
opt_names = list(results.keys())
accuracies = [results[opt]['test_accuracy'] for opt in opt_names]
plt.bar(opt_names, accuracies, color=['blue', 'orange', 'green', 'red', 'purple'])
plt.title("Test Accuracy by Optimizer")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)
for i, v in enumerate(accuracies):
    plt.text(i, v + 1, f"{v:.1f}%", ha='center', fontweight='bold')

plt.tight_layout()
plt.show()



import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import os
import csv
import random
import numpy as np
from collections import defaultdict
from torchvision.transforms import autoaugment


# ========== 设置随机种子 ==========
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

os.makedirs("results/figures", exist_ok=True)
os.makedirs("results", exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Using device: {device}")

BATCH_SIZE = 64  # 显存充足，用大batch
IMG_SIZE = 96  # STL-10 原生分辨率
EPOCHS = 50  # 充分训练
NUM_CLASSES = 10
MAX_LR = 0.01  # OneCycle 最大学习率
NUM_WORKERS = 4
PIN_MEMORY = True


# ========== CutMix ==========
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def cutmix_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x, y, y[index], lam


# ========== 数据增强 ==========
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(96),
    transforms.RandomHorizontalFlip(),
    autoaugment.AutoAugment(autoaugment.AutoAugmentPolicy.IMAGENET),  # 👈 神级增强！
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ========== 加载数据集 ==========
print("📂 Loading STL-10 dataset...")
train_dataset = datasets.STL10(root='./data', split='train', download=True, transform=transform_train)
test_dataset = datasets.STL10(root='./data', split='test', download=True, transform=transform_test)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    persistent_workers=True if NUM_WORKERS > 0 else False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    persistent_workers=True if NUM_WORKERS > 0 else False
)


# ========== 模型定义 ==========
def get_model(num_classes=10):
    model = models.resnet50(pretrained=True)  # 👈 关键：使用预训练权重！
    # 替换分类头
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    # 初始化新层
    nn.init.kaiming_normal_(model.fc.weight)
    nn.init.constant_(model.fc.bias, 0)
    return model.to(device)


# ========== Lookahead Wrapper ==========
class Lookahead(object):
    def __init__(self, optimizer, k=5, alpha=0.5):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        self.state = defaultdict(dict)
        self.fast_state = self.optimizer.state
        for group in self.param_groups:
            group["counter"] = 0

    def update(self, group):
        for fast in group["params"]:
            param_state = self.state[fast]
            if "slow_param" not in param_state:
                param_state["slow_param"] = torch.zeros_like(fast.data)
                param_state["slow_param"].copy_(fast.data)
            slow = param_state["slow_param"]
            slow += (fast.data - slow) * self.alpha
            fast.data.copy_(slow)

    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        for group in self.param_groups:
            if group["counter"] == 0:
                self.update(group)
            group["counter"] += 1
            if group["counter"] >= self.k:
                group["counter"] = 0
        return loss


# ========== RAdam 实现 ==========
class RAdam(optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(RAdam, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1

                if group['weight_decay'] != 0:
                    p.data.add_(-group['weight_decay'] * group['lr'], p.data)

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                rho_inf = 2 / (1 - beta2) - 1
                rho_t = rho_inf - 2 * state['step'] * (beta2 ** state['step']) / (1 - beta2 ** state['step'])
                if rho_t > 4:
                    rect = (rho_t - 4) * (rho_t - 2) * rho_inf / ((rho_inf - 4) * (rho_inf - 2) * rho_t)
                    adam_step = exp_avg / (exp_avg_sq.sqrt().add_(group['eps']))
                    p.data.add_(adam_step, alpha=-group['lr'] * rect)
                else:
                    p.data.add_(exp_avg, alpha=-group['lr'])

        return loss


# ========== 优化器工厂函数（统一高性能配置）==========
def get_optimizer_and_scheduler(name, params):
    if name == "SGD":
        optimizer = optim.SGD(params, lr=MAX_LR, momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=MAX_LR, steps_per_epoch=len(train_loader),
                                                  epochs=EPOCHS)
    elif name == "SGD_Nesterov":
        optimizer = optim.SGD(params, lr=MAX_LR, momentum=0.9, nesterov=True, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=MAX_LR, steps_per_epoch=len(train_loader),
                                                  epochs=EPOCHS)
    elif name == "Adam":
        optimizer = optim.Adam(params, lr=MAX_LR / 10, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=MAX_LR / 10, steps_per_epoch=len(train_loader),
                                                  epochs=EPOCHS)
    elif name == "AdamW":
        optimizer = optim.AdamW(params, lr=MAX_LR / 10, weight_decay=1e-2)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=MAX_LR / 10, steps_per_epoch=len(train_loader),
                                                  epochs=EPOCHS)
    elif name == "RMSprop":
        optimizer = optim.RMSprop(params, lr=MAX_LR, alpha=0.99, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=MAX_LR, steps_per_epoch=len(train_loader),
                                                  epochs=EPOCHS)
    elif name == "Adagrad":
        optimizer = optim.Adagrad(params, lr=MAX_LR / 10, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    elif name == "Adadelta":
        optimizer = optim.Adadelta(params, lr=1.0, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    elif name == "NAdam":
        optimizer = optim.NAdam(params, lr=MAX_LR / 10, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=MAX_LR / 10, steps_per_epoch=len(train_loader),
                                                  epochs=EPOCHS)
    elif name == "Adamax":
        optimizer = optim.Adamax(params, lr=MAX_LR / 5, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=MAX_LR / 5, steps_per_epoch=len(train_loader),
                                                  epochs=EPOCHS)

    elif name == "RAdam":
        optimizer = RAdam(params, lr=MAX_LR / 10, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=MAX_LR / 10, steps_per_epoch=len(train_loader),
                                                  epochs=EPOCHS)
    else:
        raise ValueError(f"Unknown optimizer: {name}")

    return optimizer, scheduler


optimizers_list = [
    "RAdam",
    "SGD",
    "SGD_Nesterov",
    "Adam",
    "AdamW",
    "RMSprop",
    "Adagrad",
    "Adadelta",
    "NAdam",
    "Adamax"
]

# ========== 混合精度训练 ==========
scaler = torch.cuda.amp.GradScaler()
criterion = nn.CrossEntropyLoss()


# ========== 训练函数（启用 CutMix）==========
def train_model(model, optimizer, scheduler, train_loader, epochs=EPOCHS):
    model.train()
    losses = []
    for epoch in range(epochs):
        total_loss = 0.0
        start_time = time.time()
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # 👇 启用 CutMix（每 batch 50% 概率）
            if random.random() < 0.5:
                images, targets_a, targets_b, lam = cutmix_data(images, labels, alpha=1.0)
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
            else:
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        epoch_time = time.time() - start_time
        current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else MAX_LR
        print(f"  Epoch [{epoch + 1}/{epochs}] Loss: {avg_loss:.4f} | LR: {current_lr:.1e} | Time: {epoch_time:.1f}s")

    return losses


# ========== 评估函数 ==========
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"  ✅ Test Accuracy: {accuracy:.2f}%")
    return accuracy


# ========== 主执行函数 ==========
def run_experiment():
    print("\n" + "=" * 80)
    print("💥 STARTING PROFESSIONAL GRADE OPTIMIZER EXPERIMENT!")
    print("=" * 80)

    results = {}

    for opt_name in optimizers_list:
        print(f"\n▶️  Training with {opt_name} optimizer...")

        model = get_model(NUM_CLASSES)
        try:
            optimizer, scheduler = get_optimizer_and_scheduler(opt_name, model.parameters())
        except Exception as e:
            print(f"⚠️ 初始化失败: {e}")
            continue

        start_time = time.time()
        train_losses = train_model(model, optimizer, scheduler, train_loader, EPOCHS)
        test_acc = evaluate_model(model, test_loader)
        total_time = time.time() - start_time

        results[opt_name] = {
            'train_losses': train_losses,
            'test_accuracy': test_acc,
            'total_train_time_sec': total_time,
            'time_per_epoch': total_time / EPOCHS,
            'final_loss': train_losses[-1]
        }

        del model, optimizer, scheduler
        torch.cuda.empty_cache()

    # ========== 可视化并保存图表 ==========
    plt.figure(figsize=(18, 6))

    # 图1：损失曲线（前6名）
    plt.subplot(1, 3, 1)
    sorted_results = sorted(results.items(), key=lambda x: x[1]['test_accuracy'], reverse=True)
    top6 = sorted_results[:6]
    for opt_name, res in top6:
        plt.plot(res['train_losses'], label=f"{opt_name} ({res['test_accuracy']:.1f}%)")
    plt.title("Training Loss (Top 6 Optimizers)", fontsize=12, fontweight='bold')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.6)

    # 图2：准确率柱状图
    plt.subplot(1, 3, 2)
    opt_names = [item[0] for item in sorted_results]
    accuracies = [item[1]['test_accuracy'] for item in sorted_results]
    colors = plt.cm.tab20.colors[:len(opt_names)]
    bars = plt.bar(range(len(opt_names)), accuracies, color=colors, edgecolor='black', linewidth=0.5)
    plt.title("Test Accuracy Ranking", fontsize=12, fontweight='bold')
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.xticks(range(len(opt_names)), [name.replace('_', '\n') for name in opt_names], rotation=45, ha='right',
               fontsize=8)
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        plt.text(i, bar.get_height() + 1, f'{acc:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

    # 图3：训练速度
    plt.subplot(1, 3, 3)
    times = [item[1]['time_per_epoch'] for item in sorted_results]
    bars2 = plt.bar(range(len(opt_names)), times, color=plt.cm.Set3.colors[:len(opt_names)], edgecolor='black',
                    linewidth=0.5)
    plt.title("Training Speed (s/epoch)", fontsize=12, fontweight='bold')
    plt.ylabel("Time (s)")
    plt.xticks(range(len(opt_names)), [name.replace('_', '\n') for name in opt_names], rotation=45, ha='right',
               fontsize=8)
    for i, (bar, t) in enumerate(zip(bars2, times)):
        plt.text(i, bar.get_height() + 0.2, f'{t:.1f}s', ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()
    fig_path = "results/figures/optimizer_comparison_pro.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.show()

    # ========== 保存 CSV ==========
    csv_path = "results/summary_results.csv"
    with open(csv_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Rank", "Optimizer", "Test Accuracy (%)", "Total Time (s)", "Time/Epoch (s)", "Final Loss"])
        for rank, (opt_name, res) in enumerate(sorted_results, 1):
            writer.writerow([
                rank,
                opt_name,
                f"{res['test_accuracy']:.2f}",
                f"{res['total_train_time_sec']:.1f}",
                f"{res['time_per_epoch']:.1f}",
                f"{res['final_loss']:.4f}"
            ])

    # ========== 保存报告 ==========
    report_path = "results/experiment_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("📊 PROFESSIONAL OPTIMIZER COMPARISON REPORT\n")
        f.write(f"📅 Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"🖼️ Resolution: {IMG_SIZE}x{IMG_SIZE} | Batch: {BATCH_SIZE} | Epochs: {EPOCHS}\n")
        f.write("=" * 60 + "\n\n🏆 TOP 3:\n")
        for i, (opt_name, res) in enumerate(sorted_results[:3]):
            medal = ["🥇", "🥈", "🥉"][i]
            f.write(
                f"  {medal} {opt_name:<15} → {res['test_accuracy']:>6.2f}% | {res['time_per_epoch']:>4.1f}s/epoch\n")

    print(f"\n✅ 图表保存至: {fig_path}")
    print(f"✅ CSV保存至: {csv_path}")
    print(f"✅ 报告保存至: {report_path}")


# ========== 程序入口 ==========
if __name__ == '__main__':
    run_experiment()



