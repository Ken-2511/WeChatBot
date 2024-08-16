# 这个里面定义了一些AI相关的工具函数
import os
import cv2
import time
import numpy as np
import urllib.parse
import subprocess as sp

__all__ = ["func_send", "func_wait", "send_message"]

# 配置临时存放文件的路径
img_dir = os.path.join(os.path.dirname(__file__), "images")
os.makedirs(img_dir, exist_ok=True)

# 配置一些按钮的位置
btn_pos = {
    "wechat": [660, 2150],
    "1st_message": [540, 400],
    "chats": [135, 2215],
    "clear_memory": [540, 2130],
}

# 手势的位置
gesture_pos = {
    # 从左到右分别是起点x, 起点y, 终点x, 终点y, 时间
    "home": [540, 2335, 540, 1900, 100],
    "back": [5, 1500, 300, 1500, 100],
    "background": [540, 1900, 540, 1900, 300],
}

# 图片的路径
img_path_map = {
    "screen": os.path.join(img_dir, "screen.png"),
    "wechat": os.path.join(img_dir, "wechat.png"),
    "send": os.path.join(img_dir, "send.png"),
    "chats": os.path.join(img_dir, "chats.png"),
}

# 所有可能的模式，用于和当前屏幕内容进行匹配
# 这里面的string可以作为key放到img_path_map中
# 大部分page都有mask，小部分可能没有
all_pages = {
    "chats_page": {
        "target": os.path.join(img_dir, "chats_page.png"),
        "mask": os.path.join(img_dir, "chats_page_mask.png"),
        "thresh": 0.03,
    },
    "keyboard_on": {
        "target": os.path.join(img_dir, "keyboard_on.png"),
        "mask": os.path.join(img_dir, "keyboard_on_mask.png"),
        "thresh": 0.05,
    },
    "home_screen": {
        "target": os.path.join(img_dir, "home_screen.png"),
        "mask": os.path.join(img_dir, "home_screen_mask.png"),
        "thresh": 0.05,
    },
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


def _check_difference(screen_img, target_img, mask_img=None, threshold=0.05):
    # check the difference between the screen image and the target image
    # the screen_img and the target_img should be in BGR format
    # the mask_img: bright pixel means the region of interest
    assert screen_img.shape == target_img.shape == (2340, 1080, 3)
    assert mask_img is None or mask_img.shape == (2340, 1080, 3)
    if mask_img is None:
        mask_img = np.ones((2340, 1080, 3), dtype=np.uint8) * 255
    # calculate the difference
    diff = np.abs(screen_img - target_img) * mask_img
    diff = diff.astype(np.float32).sum().item() / mask_img.sum().item()
    # print(f"Difference: {diff}")  # for debug purpose
    return diff < threshold


def check_page(take_screenshot=False):
    # check if the screen matches any of the predefined pages
    # if matched, return the one; if not, return None
    if take_screenshot:
        _get_screen()
    screen_img = cv2.imread(img_path_map["screen"], cv2.IMREAD_COLOR)
    for key, page in all_pages.items():
        target_img = cv2.imread(page["target"], cv2.IMREAD_COLOR)
        mask_img = cv2.imread(page["mask"], cv2.IMREAD_COLOR)
        thresh = page["thresh"]
        result = _check_difference(screen_img, target_img, mask_img, thresh)
        if result:
            return key
    return None


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


def send_message(message):
    # default send to the current chat
    # make sure that the chat is opened
    _input_text(message)


def start_wechat():
    _swipe(*gesture_pos["home"], wait=200)
    _swipe(*gesture_pos["home"])
    _tap(*btn_pos["wechat"])
    current_page = check_page(take_screenshot=True)
    print(current_page)
    count = 0
    while current_page != "chats_page":
        if current_page is None:
            print("Unknown page")
            # clear the memory
            _swipe(*gesture_pos["home"])
            sp.run("adb shell input keyevent KEYCODE_MENU", shell=True)
            time.sleep(1)
            _tap(*btn_pos["clear_memory"])
            _tap(*btn_pos["wechat"])
            time.sleep(count + 2)
            current_page = check_page(take_screenshot=True)


def _get_img_rela_pos(upper_img, lower_img, reserve_height=500):
    # 这两张图片应当是聊天界面的截图（裁剪好了的）
    # 返回的是下一张图片在上一张图片中的相对位置
    # 保证会返回一个值，即使匹配度不高，也返回那个最大的值
    # 假定两张图片重叠的高度不会小于reserve_height
    assert upper_img.shape == lower_img.shape
    upper_img = upper_img.astype(np.float32)
    lower_img = lower_img.astype(np.float32)
    h, w, _ = upper_img.shape
    _proposed_dh = 0
    _proposed_diff = 1000
    # 先粗略地找到重叠的区域
    for dh in range(0, h - reserve_height, 10):
        upper = upper_img[dh:]
        lower = lower_img[:h - dh]
        diff = np.abs(upper - lower).mean()
        if diff < _proposed_diff:
            _proposed_dh = dh
            _proposed_diff = diff
    # 精细地找到重叠的区域
    _dh = 0
    _diff = 1000
    lower_bound = max(0, _proposed_dh - 30)
    upper_bound = min(h - reserve_height, _proposed_dh + 30)
    for dh in range(lower_bound, upper_bound, 1):
        upper = upper_img[dh:]
        lower = lower_img[:h - dh]
        diff = np.abs(upper - lower).mean()
        if diff < _diff:
            _dh = dh
            _diff = diff
    if _diff > 15:
        print(f"Warning: The difference is {_diff}")
    return _dh


def long_screenshot(height=2000):
    # long screenshot
    current_height = 0
    sp.run("adb shell screencap -p /sdcard/screen.png", shell=True)
    sp.run(f"adb pull /sdcard/screen.png {os.path.join(img_dir, 'long_screen', f"{0}.png")}", shell=True)
    count = 1
    dhs = []
    while current_height < height:
        _swipe(540, 600, 540, 1780, 1000)
        sp.run("adb shell screencap -p /sdcard/screen.png", shell=True)
        sp.run(f"adb pull /sdcard/screen.png {os.path.join(img_dir, 'long_screen', f"{count}.png")}", shell=True)
        # calculate the relative difference
        upper_img = cv2.imread(os.path.join(img_dir, "long_screen", f"{count}.png"), cv2.IMREAD_COLOR)[230:2130]
        lower_img = cv2.imread(os.path.join(img_dir, "long_screen", f"{count - 1}.png"), cv2.IMREAD_COLOR)[230:2130]
        dh = _get_img_rela_pos(upper_img, lower_img)
        dhs.append(dh)
        print(f"dh: {dh}")
        if dh == 0:
            break
        count += 1
    # 拼接图片
    total_height = sum(dhs) + 2130 - 230
    total_img = np.zeros((total_height, 1080, 3), dtype=np.uint8)
    for i in range(count):
        img = cv2.imread(os.path.join(img_dir, "long_screen", f"{i}.png"), cv2.IMREAD_COLOR)[230:2130]
        print(sum(dhs[:i]), sum(dhs[:i + 1]))
        total_img[total_height - sum(dhs[:i]) - 2130 + 230:total_height - sum(dhs[:i]), :] = img
    cv2.imwrite(os.path.join(img_dir, "long_screen.png"), total_img)


if __name__ == '__main__':
    long_screenshot()
    # img1 = cv2.imread(os.path.join(img_dir, "long_screen.png"), cv2.IMREAD_COLOR)[230:2130]
    # img2 = cv2.imread(os.path.join(img_dir, "long_screen1.png"), cv2.IMREAD_COLOR)[230:2130]
    # print(_get_img_rela_pos(img1, img2))
    # _get_screen()
    # start_wechat()
    # print(check_page())
    # print(check_if_keyboard_on())
    # img = cv2.imread(img_path_map["screen"], cv2.IMREAD_COLOR)
    # img1 = cv2.imread(os.path.join(img_dir, "screen1.png"), cv2.IMREAD_COLOR)
    # img2 = np.ones((2340, 1080, 3), dtype=np.uint8) * 255
    # mask = cv2.imread(r"C:\Users\IWMAI\Desktop\screen.png", cv2.IMREAD_COLOR)
    # _check_difference(img, img)
    # _check_difference(img, img1)
    # _check_difference(img, img2)
    # _check_difference(img, img, mask)
    # _check_difference(img, img1, mask)
    # _check_difference(img, img2, mask)
