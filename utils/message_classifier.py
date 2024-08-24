# 训练一个AI来检测一张消息截图的类型（文字，图片，表情包……）
import os
import cv2
import json
import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

__all__ = ["MessageClassifier"]

file_dir = os.path.dirname(__file__)

# 所有消息类型
all_types = ["text", "system", "sticker", "image", "tickle", "file", "voice", "voice/video call", "mini-program", "other"]
_weights = {
    "text": 1,
    "system": 2,
    "sticker": 3,
    "image": 4,
    "tickle": 1,
    "file": 6,
    "voice": 6,
    "voice/video call": 6,
    "mini-program": 6,
    "other": 3
}
_ = sum(_weights.values())
type_weights = torch.tensor([_weights[t] / _ for t in all_types], dtype=torch.float32)


class MessageClassifier(nn.Module):
    def __init__(self, pretrained=False):
        super(MessageClassifier, self).__init__()
        self.model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        self.model.features.requires_grad_(False)  # freeze the feature extractor
        self.model.classifier = nn.Sequential(
            # nn.Dropout(p=0.2),
            nn.Linear(in_features=1280, out_features=128),
            nn.ReLU(),
            # nn.Dropout(p=0.2),
            nn.Linear(in_features=128, out_features=len(all_types))
        )
        if pretrained:
            path = os.path.join(file_dir, "train_data", "checkpoints", "epoch50.pt")
            self.load_state_dict(torch.load(path, weights_only=True))

    def forward(self, x):
        return self.model(x)

    def predict_numpy(self, x):
        # 作为对外的接口，接受numpy数组，返回预测结果
        # 裁剪和缩放图片（一般来讲图片宽度741，将图片缩放到224x224）
        h, w, _ = x.shape
        new_h, new_w = 224 * h // w, 224
        x = cv2.resize(x, (new_w, new_h))
        new_x = np.ones((224, 224, 3), dtype=np.uint8) * 237
        if new_h > 224:
            new_x = x[:224, :]
        else:
            new_x[:new_h, :] = x[:, :]
        x = new_x.astype(np.float32)
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)
        output = self(x)
        _, predicted = torch.max(output, 1)
        return all_types[predicted.item()]


class MyDataset(Dataset):
    def __init__(self, file_path):
        self.json_file = file_path
        with open(self.json_file, "r") as file:
            self.json_file = json.load(file)
        self.data = []
        self.target = []
        for key, value in self.json_file.items():
            self.data.append(key)
            self.target.append(value)
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # read image
        img = cv2.imread(f"train_data/msg_images/{self.data[idx]}", cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # pad the image, always maintain the width and pad the height,
        # paste the image on the top of the padded image
        h, w, _ = img.shape
        new_image = np.ones((w, w, 3), dtype=np.uint8) * 237
        if h > w:
            new_image[:, :] = img[:w, :]
        else:
            new_image[:h, :] = img[:, :]
        img = cv2.resize(img, (224, 224))
        img = self.transform(img)
        # process target
        target = all_types.index(self.target[idx])
        return img, target


def train(model, train_loader, val_loader, epochs=10, lr=0.001, device="cuda"):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=type_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

        model.eval()
        total = 0
        correct = 0
        total_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                output = model(x)
                loss = criterion(output, y)
                total_loss += loss.item()
                _, predicted = torch.max(output, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        total_loss /= total

        # update the lr
        lr *= 0.95
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        # print the result
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}, Accuracy: {correct / total:.4f}, "
              f"Val Loss: {total_loss:.4f}")
        # save model
        torch.save(model.state_dict(), f"train_data/checkpoints/epoch{epoch + 1}.pt")


def show_results(model, val_loader):
    # 随机抽取图片，用cv2显示图片，然后显示预测结果和置信度
    while True:
        idx = np.random.randint(len(val_loader.dataset))
        x, y = val_loader.dataset[idx]
        x = x.unsqueeze(0)
        output = model(x)
        _, predicted = torch.max(output, 1)
        confidence = torch.softmax(output, 1).max().item()
        img = cv2.imread(f"train_data/msg_images/{val_loader.dataset.data[idx]}", cv2.IMREAD_COLOR)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imshow("Image", img)
        print(f"Predicted: {all_types[predicted.item()]}, Confidence: {confidence}")
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break


def rename_files():
    # 为了防止复制进来的文件和已有的文件重名，我们需要重命名文件
    # 原始的文件名是一个数字，我们将其变成负数并且递减

    # 找到所有未正确命名的文件
    all_files = os.listdir("train_data/msg_images")
    positive_files = [f for f in all_files if int(f.split(".")[0]) >= 0]
    negative_files = [f for f in all_files if int(f.split(".")[0]) < 0]
    start_index = min([int(f.split(".")[0]) for f in negative_files]) - 1 if negative_files else -1

    # 重命名文件
    for i, f in enumerate(positive_files):
        os.rename(f"train_data/msg_images/{f}", f"train_data/msg_images/{start_index - i}.png")


def label_data():
    import tkinter as tk
    from tkinter import messagebox
    from PIL import Image, ImageTk
    # 指定json文件和图片文件夹
    json_file = "train_data/labels.json"
    img_folder = "train_data/msg_images"

    # 重命名文件
    rename_files()

    # 加载已有的JSON文件，如果不存在则创建一个空字典
    if os.path.exists(json_file):
        with open(json_file, "r") as f:
            data = json.load(f)
    else:
        data = {}

    # 找到未标记的图片
    images = os.listdir(img_folder)
    images_to_label = [img for img in images if img not in data]

    if not images_to_label:
        print("所有图片都已经标记完成。")
        return

    # 创建Tkinter窗口
    root = tk.Tk()
    root.title("Image Labeling Tool")

    # 记录已标记的图片和标签的堆栈，用于撤回
    history_stack = []

    def pad_and_resize_image(image, target_width=300):
        """将图片填充为正方形并缩放到指定宽度"""
        h, w, _ = image.shape
        if h > w:
            pad = (h - w) // 2
            padded_image = cv2.copyMakeBorder(image, 0, 0, pad, h - w - pad, cv2.BORDER_CONSTANT, value=[237, 237, 237])
        else:
            pad = (w - h) // 2
            padded_image = cv2.copyMakeBorder(image, pad, w - h - pad, 0, 0, cv2.BORDER_CONSTANT, value=[237, 237, 237])

        resized_image = cv2.resize(padded_image, (target_width, target_width))
        return resized_image

    # 加载并显示图片
    def show_image(img_path):
        image = cv2.imread(img_path)
        image = pad_and_resize_image(image, target_width=300)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)
        panel.configure(image=image)
        panel.image = image

    # 标记当前图片
    def label_current_image(label):
        current_img = images_to_label[0]
        # 将当前图片和标签存入堆栈
        history_stack.append((current_img, label))
        data[current_img] = label
        images_to_label.pop(0)  # 移除已标记的图片
        if images_to_label:
            show_image(os.path.join(img_folder, images_to_label[0]))
        else:
            messagebox.showinfo("完成", "所有图片都已标记完成。")
            save_and_exit()

    # 撤回到上一个标记
    def undo_last_label():
        if history_stack:
            last_image, last_label = history_stack.pop()  # 从堆栈中弹出最后一个
            images_to_label.insert(0, last_image)  # 将其重新插入到待标记列表的开头
            del data[last_image]  # 删除已保存的标记
            show_image(os.path.join(img_folder, last_image))
        else:
            messagebox.showinfo("提示", "没有可以撤回的标记。")

    # 保存并退出
    def save_and_exit():
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=4)
        root.destroy()

    # 创建主框架，水平排列图片和按钮
    main_frame = tk.Frame(root)
    main_frame.pack(padx=10, pady=10)

    # 创建用于放置图片的框架
    image_frame = tk.Frame(main_frame)
    image_frame.pack(side="left")

    # 创建用于放置按钮的框架
    button_frame = tk.Frame(main_frame)
    button_frame.pack(side="right", padx=10)

    # 放置图片的Label
    panel = tk.Label(image_frame)
    panel.pack(padx=10, pady=10)

    # 创建标签按钮
    labels = all_types
    for label in labels:
        btn = tk.Button(button_frame, text=label, command=lambda l=label: label_current_image(l), width=15)
        btn.pack(pady=5)  # 竖直排列按钮

    # 创建撤回按钮
    undo_button = tk.Button(button_frame, text="撤回", command=undo_last_label, width=15, fg="red")
    undo_button.pack(pady=20)  # 增加一些额外的空隙以与其他按钮分开

    # 显示第一张图片
    show_image(os.path.join(img_folder, images_to_label[0]))

    root.protocol("WM_DELETE_WINDOW", save_and_exit)
    root.mainloop()


def print_label_stats():
    # 读取JSON文件
    json_file = "train_data/labels.json"
    with open(json_file, 'r') as f:
        data = json.load(f)
    # 初始化一个字典来统计每种类型的数量
    type_count = {msg_type: 0 for msg_type in all_types}
    # 统计每种类型的样本数量
    total = 0
    for key, value in data.items():
        if value in type_count:
            type_count[value] += 1
            total += 1
    # 打印结果
    for msg_type in all_types:
        print(f"{msg_type}: {type_count[msg_type]} 个样本，占比 {type_count[msg_type] / total:.2%}")


if __name__ == '__main__':
    # label_data()
    # split_train_val()
    # print_label_stats()
    # model = MessageClassifier()
    # train_dataset = MyDataset("train_data/train_labels.json")
    # train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    # val_dataset = MyDataset("train_data/val_labels.json")
    # val_loader = DataLoader(val_dataset, batch_size=16)
    # train(model, train_loader, val_loader, epochs=50, device="cuda")

    model = MessageClassifier()
    model.load_state_dict(torch.load("train_data/checkpoints/epoch50.pt", weights_only=True))
    model.eval()
    val_dataset = MyDataset("train_data/val_labels.json")
    val_loader = DataLoader(val_dataset, batch_size=1)
    show_results(model, val_loader)
