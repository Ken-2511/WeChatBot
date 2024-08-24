# 这是主程序，长期运行，负责一直监控微信然后做出对应的对话

import os
import cv2
import time
import shutil
from utils import phone_control as pc
from utils import message_control as mc
from utils import AI

file_dir = os.path.dirname(__file__)
utils_dir = os.path.join(file_dir, "utils")
# 一些常用配置
LOOP_INTERVAL = 1  # 循环间隔，单位秒


def initialize_chat() -> mc.Chat:
    # prerequisite: 当前手机的界面是一个新的聊天
    img = pc.get_screen()
    other_name = pc.get_chat_name(img)
    chat = mc.Chat(other_name)
    pc.long_screenshot(10)
    long_img = cv2.imread(os.path.join(utils_dir, "images", "long_screen.png"), cv2.IMREAD_COLOR)
    pc.split_chat_img(long_img, target_dir=chat.msg_img_dir)
    return chat


def __remove_chat(chat: mc.Chat or str):
    # remove the chat from the chat list
    if isinstance(chat, mc.Chat):
        chat = chat.other_id
    shutil.rmtree(os.path.join(mc.project_dir, "chats", chat))


def add_message(chat: mc.Chat, img, author):
    pass


def main_loop():
    while True:
        screen_img = pc.get_screen()
        if pc.check_page(False) != "chats_page":
            pc.start_wechat()
        time.sleep(LOOP_INTERVAL)


def test():
    initialize_chat()


if __name__ == '__main__':
    initialize_chat()
