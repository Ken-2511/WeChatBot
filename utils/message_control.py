import os
import cv2
from AI import AI
import numpy as np
from enum import Enum
from typing import Literal

platform = "windows"
file_dir = os.path.dirname(__file__)
project_dir = os.path.dirname(file_dir)


class Message:
    def __init__(self, dtype="text"):
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
    def __init__(self, other_id):
        self.messages = []
        self.other_id = other_id
        os.mkdir(os.path.join(project_dir, "chats", other_id))

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
        # split the long image into several chat images by detecting the icon
        # each chat image should only contain one message
        # return the list of chat images
        ICON_SIZE = 108
        MY_X = 940
        OTHER_X = 32
        MESSAGE_X1 = 169
        MESSAGE_X2 = 910
        THRESH = 0.85
        assert my_icon.shape == other_icon.shape == (ICON_SIZE, ICON_SIZE, 3)
        assert long_img.shape[1] == 1080
        chat_name = self.get_chat_name(long_img)
        chats = []
        h, w, _ = long_img.shape
        # use matchTemplate to find the positions of the icons for myself
        my_icon_pos = []
        temp_img = long_img[:, MY_X:MY_X + ICON_SIZE]
        res = cv2.matchTemplate(temp_img, my_icon, cv2.TM_CCOEFF_NORMED).reshape(-1)
        for i in range(len(res)):
            if res[i] > THRESH:
                b0, b1 = max(0, i - 5), min(len(res), i + 5)
                idx = b0 + np.argmax(res[b0:b1]).item()
                my_icon_pos.append(idx)
        # use matchTemplate to find the positions of the icons for the other
        other_icon_pos = []
        temp_img = long_img[:, OTHER_X:OTHER_X + ICON_SIZE]
        res = cv2.matchTemplate(temp_img, other_icon, cv2.TM_CCOEFF_NORMED).reshape(-1)
        i = 0
        while i < len(res):
            if res[i] > THRESH:
                b0, b1 = max(0, i - 5), min(len(res), i + 5)
                idx = b0 + np.argmax(res[b0:b1]).item()
                other_icon_pos.append(idx)
                i = idx + ICON_SIZE
            i += 1
        # split the chat images
        ci_dir = os.path.join(img_dir, "chat_images")
        os.makedirs(ci_dir, exist_ok=True)
        icon_pos = sorted(my_icon_pos + other_icon_pos)
        long_img = long_img[:, MESSAGE_X1:MESSAGE_X2]
        time_img = cv2.imread(os.path.join(img_dir, "time_img.png"), cv2.IMREAD_COLOR)
        assert time_img.shape == (153, 741, 3), time_img.shape
        for i in range(len(icon_pos) - 1):
            chat_img = long_img[icon_pos[i]:icon_pos[i + 1] - 32]
            if (chat_img.shape[0] > 153 and
                    _check_difference(chat_img[chat_img.shape[0] - 153:], time_img, threshold=0.08, check_shape=False)):
                chat_img = chat_img[:chat_img.shape[0] - 153]
            cv2.imwrite(os.path.join(ci_dir, f"{i}.png"), chat_img)
            chats.append(chat_img)
        return chats


if __name__ == '__main__':
    # for i in range(90):
    img = cv2.imread(f"images/chat_images/{16}.png")
    ans = AI.detect_unknown(img)
    print(ans)