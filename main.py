import os
import cv2
import shutil
from utils import phone_control as pc
from utils import message_control as mc
from utils import AI
from utils.phone_control import split_chat_img

file_dir = os.path.dirname(__file__)
utils_dir = os.path.join(file_dir, "utils")


def initialize_chat() -> mc.Chat:
    # prerequisite: 当前手机的界面是一个新的聊天
    img = pc.get_screen()
    other_name = pc.get_chat_name(img)
    chat = mc.Chat(other_name)
    pc.long_screenshot(10)
    long_img = cv2.imread(os.path.join(utils_dir, "images", "long_screen.png"), cv2.IMREAD_COLOR)
    img, data = pc.split_chat_img(long_img, chat.my_icon, chat.other_icon, target_dir=chat.msg_img_dir)
    print(data)
    return chat


def __remove_chat(chat: mc.Chat or str):
    # remove the chat from the chat list
    if isinstance(chat, mc.Chat):
        chat = chat.other_id
    shutil.rmtree(os.path.join(mc.project_dir, "chats", chat))


def add_message(chat: mc.Chat, img, author):
    pass


def test():
    initialize_chat()


if __name__ == '__main__':
    test()
    # __remove_chat("Yongkang CE")