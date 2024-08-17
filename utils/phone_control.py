# 这个里面定义了一些AI相关的工具函数
import os
import cv2
import time
import base64
from utils.AI import AI
import numpy as np
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


def get_screen():
    # show the screen using adb
    sp.run("adb shell screencap -p /sdcard/screen.png", shell=True)
    sp.run(f"adb pull /sdcard/screen.png {img_dir}", shell=True)
    img = cv2.imread(img_path_map["screen"], cv2.IMREAD_COLOR)
    return img


def _input_text(text):
    # input text using adb
    text = str(base64.b64encode(text.encode("utf-8")))[1:]
    sp.run(f"adb shell am broadcast -a ADB_INPUT_B64 --es msg {text}", shell=True)


def __tap_img(img_key, wait=1000):
    # 暂时废弃
    # read the image and get the newest screen
    img_path = img_path_map[img_key]
    target_img = cv2.imread(img_path)
    get_screen()
    screen_img = cv2.imread(os.path.join(img_dir, "screen.png"))
    # find the target image in the screen image
    res = cv2.matchTemplate(screen_img, target_img, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    h, w, _ = target_img.shape
    x, y = max_loc
    _tap(x + w // 2, y + h // 2, wait)
    print(f"Tap {img_key} at ({x + w // 2}, {y + h // 2})")  # for debug purpose


def _check_difference(screen_img, target_img, mask_img=None, threshold=0.05, check_shape=True):
    # check the difference between the screen image and the target image
    # the screen_img and the target_img should be in BGR format
    # the mask_img: bright pixel means the region of interest
    if check_shape:
        assert screen_img.shape == target_img.shape == (2340, 1080, 3)
        assert mask_img is None or mask_img.shape == (2340, 1080, 3)
    if mask_img is None:
        mask_img = np.ones(screen_img.shape, dtype=np.uint8) * 255
    # calculate the difference
    diff = np.abs(screen_img - target_img) * mask_img
    diff = diff.astype(np.float32).sum().item() / mask_img.sum().item()
    # print(f"Difference: {diff}")  # for debug purpose
    return diff < threshold


def check_page(take_screenshot=False):
    # check if the screen matches any of the predefined pages
    # if matched, return the one; if not, return None
    if take_screenshot:
        get_screen()
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


def _get_img_rela_pos(upper_img, lower_img, reserve_height=500, gap=10):
    # 这两张图片应当是聊天界面的截图（裁剪好了的）
    # 返回的是下一张图片在上一张图片中的相对位置
    # 保证会返回一个值，即使匹配度不高，也返回那个最大的值
    # 假定两张图片重叠的高度不会小于reserve_height
    assert upper_img.shape[1] == lower_img.shape[1]
    assert upper_img.shape[0] > reserve_height and lower_img.shape[0] > reserve_height
    # 横向压缩图片大小
    upper_img = upper_img[:, ::5, :]
    lower_img = lower_img[:, ::5, :]
    # 将图片转换为float32类型
    upper_img = upper_img.astype(np.float32)
    lower_img = lower_img.astype(np.float32)
    h_u, w, _ = upper_img.shape
    h_l = lower_img.shape[0]
    h = min(h_u, h_l)
    _proposed_dh = 0
    _proposed_diff = 1000
    # 先粗略地找到重叠的区域
    for dh in range(0, h - reserve_height, gap):
        upper = upper_img[h_u - h + dh:]
        lower = lower_img[:h - dh]
        diff = np.abs(upper - lower).mean()
        # print(f"dh: {dh}, diff: {diff}")
        if diff < _proposed_diff:
            _proposed_dh = dh
            _proposed_diff = diff
    # 精细地找到重叠的区域
    _dh = 0
    _diff = 1000
    lower_bound = max(0, _proposed_dh - 30)
    upper_bound = min(h - reserve_height, _proposed_dh + 30)
    for dh in range(lower_bound, upper_bound, 1):
        upper = upper_img[h_u - h + dh:]
        lower = lower_img[:h - dh]
        diff = np.abs(upper - lower).mean()
        if diff < _diff:
            _dh = dh
            _diff = diff
    if _diff > 15:
        print(f"Warning: The difference is {_diff}")
    return _dh, _diff


def long_screenshot(count_limit=15, start_from=0):
    # long screenshot
    UP_BOUND = 230
    DOWN_BOUND = 2130
    count = start_from
    if count == 0:
        # 如果从0开始的话，就需要先截取一张屏幕，如果不是从0开始，默认刚刚已经截过图了
        sp.run("adb shell screencap -p /sdcard/screen.png", shell=True)
        sp.run(f"adb pull /sdcard/screen.png {os.path.join(img_dir, 'long_screen', f"{count}.png")}", shell=True)
    count += 1
    current_height = DOWN_BOUND - UP_BOUND
    while count < count_limit:
        _swipe(540, 600, 540, 1780, 1000)
        sp.run("adb shell screencap -p /sdcard/screen.png", shell=True)
        sp.run(f"adb pull /sdcard/screen.png {os.path.join(img_dir, 'long_screen', f"{count}.png")}", shell=True)
        # calculate the relative difference
        upper_img = cv2.imread(os.path.join(img_dir, "long_screen", f"{count}.png"), cv2.IMREAD_COLOR)[UP_BOUND:DOWN_BOUND]
        lower_img = cv2.imread(os.path.join(img_dir, "long_screen", f"{count - 1}.png"), cv2.IMREAD_COLOR)[UP_BOUND:DOWN_BOUND]
        dh, diff = _get_img_rela_pos(upper_img, lower_img)
        # dhs.append(dh)
        print(f"dh: {dh}")
        if dh == 0:
            break
        count += 1
        current_height += dh
    # 拼接图片
    dhs = []
    img0 = cv2.imread(os.path.join(img_dir, "long_screen", f"{0}.png"), cv2.IMREAD_COLOR)[UP_BOUND:DOWN_BOUND]
    for i in range(1, count):
        img1 = cv2.imread(os.path.join(img_dir, "long_screen", f"{i}.png"), cv2.IMREAD_COLOR)[UP_BOUND:DOWN_BOUND]
        dhs.append(_get_img_rela_pos(img1, img0)[0])
        img0 = img1
    total_height = sum(dhs) + DOWN_BOUND - UP_BOUND
    total_img = np.zeros((total_height, 1080, 3), dtype=np.uint8)
    for i in range(count):
        img = cv2.imread(os.path.join(img_dir, "long_screen", f"{i}.png"), cv2.IMREAD_COLOR)[UP_BOUND:DOWN_BOUND]
        total_img[total_height - sum(dhs[:i]) - DOWN_BOUND + UP_BOUND:total_height - sum(dhs[:i]), :] = img
    cv2.imwrite(os.path.join(img_dir, "long_screen.png"), total_img)


def split_chat_img(long_img, my_icon, other_icon, target_dir=None):
    # split the long image into several chat images by detecting the icon
    # each chat image should only contain one message. It is either me, or other, or system
    # return the list of chat images

    ICON_SIZE = 108
    MY_X = 940
    OTHER_X = 32
    MESSAGE_X1 = 169
    MESSAGE_X2 = 910
    THRESH = 0.85
    SYS_MSG_H = 153
    assert my_icon.shape == other_icon.shape == (ICON_SIZE, ICON_SIZE, 3)
    assert long_img.shape[1] == 1080
    h, w, _ = long_img.shape

    # make dir
    if target_dir is None:
        ci_dir = os.path.join(img_dir, "chat_images")
    else:
        ci_dir = target_dir
    os.makedirs(ci_dir, exist_ok=True)

    # get the positions of the icons for me
    my_icon_pos = []
    temp_img = long_img[:, MY_X:MY_X + ICON_SIZE]
    res = cv2.matchTemplate(temp_img, my_icon, cv2.TM_CCOEFF_NORMED).reshape(-1)
    for i in range(len(res)):
        if res[i] > THRESH:
            b0, b1 = max(0, i - 5), min(len(res), i + 5)
            idx = b0 + np.argmax(res[b0:b1]).item()
            my_icon_pos.append(idx)

    # get the positions of the icons for other
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

    # check each message to see if system message attached
    sys_pos = []
    icon_pos = sorted(my_icon_pos + other_icon_pos) + [h]
    time_img = cv2.imread(os.path.join(img_dir, "time_img.png"), cv2.IMREAD_COLOR)
    assert time_img.shape == (SYS_MSG_H, 741, 3), time_img.shape
    for i in range(len(icon_pos) - 1):
        assert icon_pos[i] < icon_pos[i + 1] - 32
        chat_img = long_img[icon_pos[i]:icon_pos[i + 1] - 32]
        _h = chat_img.shape[0]
        if chat_img.shape[0] > SYS_MSG_H:
            chat_img = chat_img[chat_img.shape[0] - SYS_MSG_H:, MESSAGE_X1:MESSAGE_X2]
            if _check_difference(chat_img, time_img, threshold=0.1, check_shape=False):
                sys_pos.append(icon_pos[i] + _h - SYS_MSG_H)

    # split the chat images and save them to the `chat_image_dir`,
    # and return the chat images, and the list of msg data
    assert len(set(my_icon_pos + other_icon_pos + sys_pos)) == len(my_icon_pos + other_icon_pos + sys_pos)
    chats = []
    msg_data = []
    icon_pos = sorted(my_icon_pos + other_icon_pos + sys_pos) + [h]
    long_img = long_img[:, MESSAGE_X1:MESSAGE_X2]
    for i in range(len(icon_pos) - 1):
        chat_img = long_img[icon_pos[i]:icon_pos[i + 1]]
        if icon_pos[i+1] not in sys_pos:
            chat_img = chat_img[:chat_img.shape[0] - 32]
        cv2.imwrite(os.path.join(ci_dir, f"{i}.png"), chat_img)
        chats.append(chat_img)
        _pos = icon_pos[i]
        author = "me" if _pos in my_icon_pos else "other" if _pos in other_icon_pos else "system"
        msg_data.append({
            "author": author,
            "f_name": f"{i}.png",
        })
    return chats, msg_data


def catch_up_chat(catch_img, count_limit=15):
    # 通过截屏的方式和之前的聊天记录衔接起来
    # catch_img是有记录的最后一页截图，我们往上滚动截图，直到找到和之前的记录重叠的部分
    for i in range(1, count_limit):
        long_screenshot(i, i-1)
        long_img = cv2.imread(os.path.join(img_dir, "long_screen.png"), cv2.IMREAD_COLOR)
        dh, diff = _get_img_rela_pos(catch_img, long_img)
        dh = min(catch_img.shape[0], long_img.shape[0]) - dh
        if diff < 15:
            long_img = long_img[dh:]
            break
    else:
        long_img = cv2.imread(os.path.join(img_dir, "long_screen.png"), cv2.IMREAD_COLOR)
    cv2.imwrite(os.path.join(img_dir, "long_screen.png"), long_img)
    return


def get_chat_name(img):
    # get the name of the chat from the chat image
    # the chat image should only contain one message
    # return the name of the chat
    assert img.shape[1] == 1080
    y1, x1, y2, x2 = 100, 120, 200, 940
    name_img = img[y1:y2, x1:x2]
    cv2.imwrite(os.path.join(img_dir, "name_img.png"), name_img)
    name = AI.get_text_from_img(name_img)
    return name


if __name__ == '__main__':
    # _input_text('hello')
    # long_screenshot()
    # start_wechat()
    # long_img = cv2.imread(os.path.join(img_dir, "long_screen.png"), cv2.IMREAD_COLOR)
    # my_icon = cv2.imread(os.path.join(img_dir, "my_icon.jpg"), cv2.IMREAD_COLOR)
    # other_icon = cv2.imread(os.path.join(img_dir, "other_icon.jpg"), cv2.IMREAD_COLOR)
    # my_icon = cv2.resize(my_icon, (108, 108))
    # other_icon = cv2.resize(other_icon, (108, 108))
    # split_chat_img(long_img, my_icon, other_icon)
    # get_screen()
    # get_chat_name(cv2.imread(os.path.join(img_dir, "screen.png"), cv2.IMREAD_COLOR))
    # import pytesseract
    # from PIL import Image
    # img = Image.open(os.path.join(img_dir, "chat_images", "16.png"))
    # pytesseract.pytesseract.tesseract_cmd = r"C:\Tesseract-OCR\tesseract.exe"
    # config = "--tessdata-dir C:/Users/IWMAI/Desktop/tessdata"
    # result = pytesseract.image_to_data(img, lang="chi_sim", config=config, output_type=pytesseract.Output.DICT)
    # # for key in result:
    # #     print(key, result[key])
    # for i in range(len(result["text"])):
    #     print(result["text"][i], result["conf"][i])
    # text = pytesseract.image_to_string(img, lang="chi_sim", config=config, output_type=pytesseract.Output.DICT)
    # print(text)
    # result = pytesseract.image_to_boxes(img, lang="chi_sim", config=config, output_type=pytesseract.Output.DICT)
    # for i in range(len(result["char"])):
    #     h = img.height
    #     print(result["char"][i], h - result["top"][i], h - result["bottom"][i])
    # _get_screen()
    # img0 = cv2.imread(os.path.join(img_dir, "screen.png"), cv2.IMREAD_COLOR)[230:2130]
    # img1 = cv2.imread(os.path.join(img_dir, "long_screen.png"), cv2.IMREAD_COLOR)
    # import time
    # t0 = time.time()
    # print(_get_img_rela_pos(img1, img0))
    # print(time.time() - t0)
    # long_screenshot(3)
    # long_screenshot(5, 2)
    UP_BOUND = 230
    DOWN_BOUND = 2130
    img = cv2.imread(os.path.join(img_dir, "screen.png"), cv2.IMREAD_COLOR)[UP_BOUND:DOWN_BOUND]
    cv2.imwrite(os.path.join(img_dir, "catch_img.png"), img)
    catch_up_chat(img)
    pass