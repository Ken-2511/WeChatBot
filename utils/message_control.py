import os
import cv2
import time
import base64
import numpy as np
from utils.AI import AI
from typing import Literal

platform = "windows"
file_dir = os.path.dirname(__file__)
project_dir = os.path.dirname(file_dir)


class Message:
    def __init__(self, img_path, my_icon, other_icon):
        # 初始化的时候直接分析出消息的类型和内容
        self.image_path = img_path
        self.chat_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        self.my_icon = my_icon
        self.other_icon = other_icon
        self.author = self.get_author()
        self.dtype = None
        self.content = None
        self.get_dtype_content()

    def json(self):
        return {
            "dtype": self.dtype,
            "author": self.author,
            "content": self.content,
        }

    def __str__(self):
        return f"Message({self.dtype}, {self.content}"

    def get_dtype_content(self):
        MESSAGE_X1 = 169  # 消息的左边界
        MESSAGE_X2 = 910  # 消息的右边界
        image = self.chat_img[:, MESSAGE_X1:MESSAGE_X2]
        mode = AI.detect_mode(image)
        # mode 有可能是 "text", "system", "sticker", "image", "tickle", "file", "voice", "voice/video call", "mini-program", "other"
        self.dtype = mode
        if mode == "text":
            self.content = AI.get_text_from_img(image)
        elif mode == "system":
            self.content = AI.get_text_from_img(image)
        elif mode == "sticker":
            self.content = AI.describe_sticker(image)
        elif mode == "image":
            self.content = AI.describe_image(image)
        elif mode == "tickle":
            self.content = "You are tickled"  # 暂且认为你不会拍别人，永远是别人拍你
        elif mode == "file":
            self.content = AI.get_file_name(image)
        elif mode == "voice":
            self.content = "voice message"  # 现阶段不做语音识别
        elif mode == "voice/video call":
            self.content = "voice/video call"
        elif mode == "mini-program":
            self.content = AI.get_text_from_img(image)
        elif mode == "other":
            self.content = AI.describe_image(image)
        else:
            raise RuntimeError(f"Unknown mode: {mode}")

    def get_author(self):
        # 分析聊天记录图片，将其分类并分析，只分析一张
        chat_img = self.chat_img

        # 1. 通过头像的位置来判断是谁发的消息
        MY_X = 940  # 我的头像的x坐标
        OTHER_X = 32  # 对方头像的x坐标
        ICON_Y = 16  # 头像的y坐标
        ICON_SIZE = 108  # 头像的大小
        MESSAGE_X1 = 169  # 消息的左边界
        MESSAGE_X2 = 910  # 消息的右边界
        THRESH = 0.8  # 匹配度的阈值
        my_icon = self.my_icon
        other_icon = self.other_icon
        assert chat_img.shape[1] == 1080
        if chat_img.shape[0] < ICON_SIZE + ICON_Y:
            # 说明这是一条系统消息
            # 提前检测的原因是防止index out of range
            return "system"
        temp_img = chat_img[ICON_Y:ICON_Y + ICON_SIZE, MY_X:MY_X + ICON_SIZE]
        res = cv2.matchTemplate(temp_img, my_icon, cv2.TM_CCOEFF_NORMED).reshape(-1)
        if max(res) > THRESH:
            return "me"
        temp_img = chat_img[ICON_Y:ICON_Y + ICON_SIZE, OTHER_X:OTHER_X + ICON_SIZE]
        res = cv2.matchTemplate(temp_img, other_icon, cv2.TM_CCOEFF_NORMED).reshape(-1)
        if max(res) > THRESH:
            return "other"
        return "system"


class Chat:
    def __init__(self, other_name):
        self.messages = []
        self.other_name = other_name
        self.other_id = base64.urlsafe_b64encode(other_name.encode('utf-8')).decode('ascii')
        self.base_dir = os.path.join(project_dir, "chats", self.other_id)
        os.makedirs(self.base_dir, exist_ok=True)
        assert os.path.exists(os.path.join(self.base_dir, "my_icon.jpg"))
        assert os.path.exists(os.path.join(self.base_dir, "other_icon.jpg"))
        self.msg_img_dir = os.path.join(self.base_dir, "msg_images")
        self.my_icon = cv2.imread(os.path.join(self.base_dir, "my_icon.jpg"))
        self.my_icon = cv2.resize(self.my_icon, (108, 108), interpolation=cv2.INTER_AREA)
        self.other_icon = cv2.imread(os.path.join(self.base_dir, "other_icon.jpg"))
        self.other_icon = cv2.resize(self.other_icon, (108, 108), interpolation=cv2.INTER_AREA)

    def add_message(self, message: Message):
        self.messages.append(message)

    def load_from_dir(self):
        pass

    def retrieve_history(self) -> bool:
        # 根据已有的消息记录，从数据库中获取历史消息
        # 因为已有的消息记录和数据库中的可能存在重叠，所以需要去重
        # 我们认为连续三条都和数据库中的重复的消息就是历史消息的结尾
        # 如果成功匹配到历史消息，返回True，否则返回False
        if len(self.messages) < 3:
            return False

    @staticmethod
    def get_chat_name(img):
        # get the name of the chat from the chat image
        # the chat image should only contain one message
        # return the name of the chat
        assert img.shape == (2340, 1080, 3)
        y1, x1, y2, x2 = 100, 120, 200, 940
        name_img = img[y1:y2, x1:x2]
        name = AI.get_text_from_img(name_img)
        return name

    def init_new_history(self, long_img, my_icon, other_icon):
        # 根据一张长长的截图，初始化聊天记录
        pass


if __name__ == '__main__':
    for i in range(40):
        my_icon = cv2.imread(r"C:\Users\IWMAI\Documents\WeChatBot\chats\WW9uZ2thbmcgQ0U=\my_icon.jpg", cv2.IMREAD_COLOR)
        other_icon = cv2.imread(r"C:\Users\IWMAI\Documents\WeChatBot\chats\WW9uZ2thbmcgQ0U=\other_icon.jpg",
                                cv2.IMREAD_COLOR)
        message = Message(rf"C:\Users\IWMAI\Documents\WeChatBot\utils\images\chat_images\{i}.png", my_icon,
                          other_icon)
        print(i, message.json())
