# 这个里面定义了一些AI相关的工具函数
import os
import cv2
import time
import urllib.parse
import subprocess as sp

__all__ = ["func_send", "func_wait"]
# 配置临时存放文件的路径
img_dir = os.path.join(os.path.dirname(__file__), "images")
os.makedirs(img_dir, exist_ok=True)
# 配置一些按钮的位置
btn_pos = {
    "wechat": [660, 2150],
    "1st_message": [540, 400],
    "chats": [135, 2215],
}
# 手势的位置
gesture_pos = {
    # 从左到右分别是起点x, 起点y, 终点x, 终点y, 时间
    "home": [540, 2335, 540, 1900, 100],
    "back": [5, 1500, 300, 1500, 100],
}
# 图片的路径
img_path_map = {
    "screen": os.path.join(img_dir, "screen.png"),
    "wechat": os.path.join(img_dir, "wechat.png"),
    "send": os.path.join(img_dir, "send.png"),
    "chats": os.path.join(img_dir, "chats.png"),
}


def func_send(content):
    print("Function Send:", content)
    return {
        "role": "tool",
        "content": "Sent"
    }


def func_wait(n):
    if n == -1:
        print("Wait indefinitely")  # for debug purpose
        time.sleep(2)
        return {
            "role": "tool",
            "content": "Waited indefinitely"
        }
    else:
        print(f"Wait for {n} seconds")  # for debug purpose
        time.sleep(2)
        return {
            "role": "tool",
            "content": f"Waited for {n} seconds"
        }


def _tap(x, y, wait=1000):
    # tap the screen using adb
    sp.run(f"adb shell input tap {x} {y}", shell=True)
    time.sleep(wait / 1000)


def _swipe(x1, y1, x2, y2, duration, wait=1000):
    # swipe the screen using adb
    sp.run(f"adb shell input swipe {x1} {y1} {x2} {y2} {duration}", shell=True)
    time.sleep(wait / 1000)


def _get_screen():
    # show the screen using adb
    sp.run("adb shell screencap -p /sdcard/screen.png", shell=True)
    sp.run(f"adb pull /sdcard/screen.png {img_dir}", shell=True)


def _input_text(text):
    # input text using adb
    text = urllib.parse.quote(text)
    sp.run(f"adb shell input text \"{text}\"", shell=True)


def _tap_img(img_key, wait=1000):
    # read the image and get the newest screen
    img_path = img_path_map[img_key]
    target_img = cv2.imread(img_path)
    _get_screen()
    screen_img = cv2.imread(os.path.join(img_dir, "screen.png"))
    # find the target image in the screen image
    res = cv2.matchTemplate(screen_img, target_img, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    h, w, _ = target_img.shape
    x, y = max_loc
    _tap(x + w // 2, y + h // 2, wait)
    print(f"Tap {img_key} at ({x + w // 2}, {y + h // 2})")  # for debug purpose


def send_message(message):
    # default send to the current chat
    # make sure that the chat is opened
    _input_text(message)


if __name__ == '__main__':
    # _swipe(*gesture_pos["home"])
    # _tap_img("wechat")
    # _tap(*btn_pos["wechat"])
    # _swipe(*gesture_pos["home"])
    # _get_screen()
    # send_message("abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz")
    # send_message("\b")
    # for i in range(1000):
    #     sp.run(f"adb shell input keyevent KEYCODE_DEL", shell=True)
    send_message("nihaoa")
    sp.run(f"adb shell input keyevent KEYCODE_SPACE", shell=True)
    send_message("xiaokeai")
    sp.run(f"adb shell input keyevent KEYCODE_SPACE", shell=True)