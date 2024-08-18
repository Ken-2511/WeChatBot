import os
import cv2
import base64
from utils.AI import AI
from typing import Literal

platform = "windows"
file_dir = os.path.dirname(__file__)
project_dir = os.path.dirname(file_dir)


class Message:
    def __init__(self, dtype="unknown"):
        self.dtype = dtype
        self.content = None
        self.id = None
        self.image_path = None
        self.json_path = None
        self.timestamp = None
        self.author = None

    def json(self):
        return {
            "dtype": self.dtype,
            "content": self.content,
            "id": self.id,
            "timestamp": self.timestamp,
            "author": self.author
        }

    def __str__(self):
        return f"Message({self.dtype}, {self.content}"

    def get_content_from_img(self, image, author: Literal["me", "other", "system"]):
        mode = AI.detect_mode(image)
        if mode == "text":
            self.content = AI.get_text_from_msg_image(image)
        elif mode == "voice":
            self.content = "voice message"
        else:  # mode == "unknown"
            self.content = AI.get_text_from_msg_image(image)
        self.author = author


class Chat:
    def __init__(self, other_name):
        self.messages = []
        self.other_name = other_name
        self.other_id = base64.urlsafe_b64encode(other_name.encode('utf-8')).decode('ascii')
        self.base_dir = os.path.join(project_dir, "chats", self.other_id)
        os.makedirs(self.base_dir, exist_ok=True)
        # assert isinstance(self.base_dir, str)
        assert os.path.exists(os.path.join(self.base_dir, "my_icon.jpg"))
        assert os.path.exists(os.path.join(self.base_dir, "other_icon.jpg"))
        self.msg_img_dir = os.path.join(self.base_dir, "msg_images")
        self.my_icon = cv2.imread(os.path.join(self.base_dir, "my_icon.jpg"))
        self.my_icon = cv2.resize(self.my_icon, (108, 108))
        self.other_icon = cv2.imread(os.path.join(self.base_dir, "other_icon.jpg"))
        self.other_icon = cv2.resize(self.other_icon, (108, 108))

    def add_message(self, message: Message):
        self.messages.append(message)

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
    # for i in range(90):
    img = cv2.imread(f"images/chat_images/{16}.png")
    ans = AI.detect_unknown(img)
    print(ans)
