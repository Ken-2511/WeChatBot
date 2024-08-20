# 这个里面定义了一些AI相关的工具函数
import os
from tkinter import image_names

import cv2
import time
import base64
from utils.AI import AI
import numpy as np
import subprocess as sp

__all__ = ["func_send", "func_wait"]

# 配置临时存放文件的路径
img_dir = os.path.join(os.path.dirname(__file__), "images")
os.makedirs(img_dir, exist_ok=True)

# 存放聊天记录的路径
chat_dir = os.path.dirname(__file__)  # utils
chat_dir = os.path.dirname(chat_dir)  # project
chat_dir = os.path.join(chat_dir, "chats")


# 配置一些按钮的位置
btn_pos = {
    "wechat": [660, 2150],
    "1st_message": [540, 400],
    "chats": [135, 2215],
    "clear_memory": [540, 2130],
    "weixin": [540, 155],
    "turn_on_keybd": [540, 2220],
    "send": [980, 2080],
}

# 手势的位置
gesture_pos = {
    # 从左到右分别是起点x, 起点y, 终点x, 终点y, 时间
    "home": [540, 2335, 540, 2000, 100],
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

# 默认的聊天黑名单——不会和这些人聊天
default_chat_blacklist = [
    "Weixin Pay",
    "Subscription ...",
    "Subscription...",
    "Minimized Groups",
]

# 所有可能的模式，用于和当前屏幕内容进行匹配
# 这里面的string可以作为key放到img_path_map中
# 大部分page都有mask，小部分可能没有
all_pages = {
    "chats_page": {
        "target": os.path.join(img_dir, "chats_page.png"),
        "mask": os.path.join(img_dir, "chats_page_mask.png"),
        "thresh": 0.03,
    },
    "chat_keybd_on": {
        "target": os.path.join(img_dir, "chat_keybd_on.png"),
        "mask": os.path.join(img_dir, "chat_keybd_on_mask.png"),
        "thresh": 0.05,
    },
    "chat_keybd_off": {
        "target": os.path.join(img_dir, "chat_keybd_off.png"),
        "mask": os.path.join(img_dir, "chat_keybd_off_mask.png"),
        "thresh": 0.05,
    },
    "home_screen": {
        "target": os.path.join(img_dir, "home_screen.png"),
        "mask": os.path.join(img_dir, "home_screen_mask.png"),
        "thresh": 0.05,
    },
}

# 用户名称和编码后名称的映射
name2enc = {
    "爸爸": "54i454i4",
    "妈妈": "5aaI5aaI",
    "Yongkang CE": "WW9uZ2thbmcgQ0U=",
    "吴雨润": "5ZC06Zuo5ram",
    "程永康Ken": "56iL5rC45bq3S2Vu",
    "Richard尹哲程": "UmljaGFyZOWwueWTsueoiw==",
    "白正督8-23〈先看朋友图)": "55m95q2j552jOC0yM-OAiOWFiOeci-aci-WPi-Wbvik=",
}
enc2name = {v: k for k, v in name2enc.items()}


def _tap(x, y, wait=1000):
    # tap the screen using adb
    sp.run(f"adb shell input tap {x} {y}", shell=True)
    time.sleep(wait / 1000)


def _db_tap(x, y, wait=1000):
    # double tap the screen using adb
    sp.run(f"adb shell input tap {x} {y}", shell=True)
    time.sleep(0.15)
    sp.run(f"adb shell input tap {x} {y}", shell=True)
    time.sleep(wait / 1000)


def _swipe(x1, y1, x2, y2, duration, wait=1000):
    # swipe the screen using adb
    sp.run(f"adb shell input swipe {x1} {y1} {x2} {y2} {duration}", shell=True)
    time.sleep(wait / 1000)


def get_screen():
    # show the screen using adb
    sp.run("adb shell screencap -p /sdcard/screen.png", shell=True)
    sp.run(f"adb pull /sdcard/screen.png {img_dir}", shell=True, stderr=sp.PIPE)
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


def func_send(content, chat_name):
    assert get_chat_name(get_screen()) == chat_name
    if check_page(False) == "chat_keybd_off":
        _tap(*btn_pos["turn_on_keybd"])
        assert check_page(True) == "chat_keybd_on"
    else:
        assert check_page(False) == "chat_keybd_on"
    _input_text(content)
    _tap(*btn_pos["send"])
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


def start_wechat():
    current_page = check_page(take_screenshot=True)
    if current_page == "chats_page":
        return
    elif current_page == "chat_keybd_on":
        _swipe(*gesture_pos["back"])
        _swipe(*gesture_pos["back"])
        start_wechat()
        return
    elif current_page == "chat_keybd_off":
        _swipe(*gesture_pos["back"])
        start_wechat()
        return
    # else——对应未知情况
    _swipe(*gesture_pos["home"], wait=200)
    _swipe(*gesture_pos["home"])
    _tap(*btn_pos["wechat"])
    current_page = check_page(take_screenshot=True)
    for count in range(5):
        if current_page == "chats_page":
            return
        # clear the memory
        _swipe(*gesture_pos["home"])
        sp.run("adb shell input keyevent KEYCODE_MENU", shell=True)
        time.sleep(1)
        _tap(*btn_pos["clear_memory"])
        _tap(*btn_pos["wechat"])
        time.sleep(count + 2)
        current_page = check_page(take_screenshot=True)
    else:
        raise Exception("Failed to start WeChat")


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


def long_screenshot(count_limit=15, start_from=0, take_screenshot=True, scroll_after=False):
    # 通过滚动截图的方式获取长截图
    # 返回的时候界面应当停留在当前页，并且已截屏
    UP_BOUND = 230
    DOWN_BOUND = 2130
    count = start_from
    current_height = DOWN_BOUND - UP_BOUND
    if take_screenshot:
        sp.run("adb shell screencap -p /sdcard/screen.png", shell=True)
        sp.run(f"adb pull /sdcard/screen.png {os.path.join(img_dir, 'long_screen', f"{count}.png")}", shell=True)
        count += 1
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
    if scroll_after:
        _swipe(540, 600, 540, 1780, 1000)
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
    SYS_MSG_H = 153 - 32
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
    i = 0
    while i < len(res):
        if res[i] > THRESH:
            b0, b1 = max(0, i - 10), min(len(res), i + 10)
            idx = b0 + np.argmax(res[b0:b1]).item()
            my_icon_pos.append(idx)
            i = idx + ICON_SIZE
        i += 1

    # get the positions of the icons for other
    other_icon_pos = []
    temp_img = long_img[:, OTHER_X:OTHER_X + ICON_SIZE]
    res = cv2.matchTemplate(temp_img, other_icon, cv2.TM_CCOEFF_NORMED).reshape(-1)
    i = 0
    while i < len(res):
        if res[i] > THRESH:
            b0, b1 = max(0, i - 10), min(len(res), i + 10)
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
        assert icon_pos[i] < icon_pos[i + 1] - 32, f"{icon_pos[i]} {icon_pos[i + 1]}"
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
        # if icon_pos[i+1] not in sys_pos:
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


def catch_up_chat(count_limit=15, take_screenshot=True):
    # 通过截屏的方式和之前的聊天记录衔接起来
    MESSAGE_X1 = 169
    MESSAGE_X2 = 910
    WIDTH = MESSAGE_X2 - MESSAGE_X1
    # 获取聊天名称
    if take_screenshot:
        get_screen()
    screen_img = cv2.imread(img_path_map["screen"], cv2.IMREAD_COLOR)
    name = name2enc.get(get_chat_name(screen_img), "unknown")
    chat_img_dir = os.path.join(chat_dir, name, "msg_images")
    # catch_img是有记录的倒数第30条至倒数第15条消息截图，我们往上滚动截图，直到找到和之前的记录重叠的部分
    img_names = sorted(os.listdir(chat_img_dir), key=lambda x: int(x.split(".")[0]))
    assert len(img_names) > 30
    img_names = img_names[-30:-15]
    imgs = [cv2.imread(os.path.join(chat_img_dir, img_name), cv2.IMREAD_COLOR) for img_name in img_names]
    assert all(img.shape[1] == WIDTH for img in imgs)
    chat_padding_img = cv2.imread(os.path.join(img_dir, "chat_padding.png"), cv2.IMREAD_COLOR)[:, MESSAGE_X1:MESSAGE_X2]
    assert chat_padding_img.shape == (32, WIDTH, 3)
    _h = sum(img.shape[0] for img in imgs) + 32 * len(imgs)
    catch_img = np.zeros((_h, WIDTH, 3), dtype=np.uint8)
    # 拼接图片
    for i, img in enumerate(imgs):
        h0 = sum(img.shape[0] for img in imgs[:i]) + 32 * i
        catch_img[h0:h0+img.shape[0]] = img
        catch_img[h0+img.shape[0]:h0+img.shape[0]+32] = chat_padding_img
    cv2.imwrite(os.path.join(img_dir, "catch_img.png"), catch_img)
    # 开始向上滚动匹配
    for i in range(1, count_limit):
        long_screenshot(i, i-1, take_screenshot=True, scroll_after=True)
        long_img = cv2.imread(os.path.join(img_dir, "long_screen.png"), cv2.IMREAD_COLOR)[:, MESSAGE_X1:MESSAGE_X2]
        dh, diff = _get_img_rela_pos(catch_img, long_img)
        dh = min(catch_img.shape[0], long_img.shape[0]) - dh
        if diff < 10:
            long_img = cv2.imread(os.path.join(img_dir, "long_screen.png"), cv2.IMREAD_COLOR)
            long_img = long_img[dh:]
            break
    else:
        long_img = cv2.imread(os.path.join(img_dir, "long_screen.png"), cv2.IMREAD_COLOR)
    cv2.imwrite(os.path.join(img_dir, "long_screen.png"), long_img)

    # 将新增的聊天记录分割开来
    long_img = cv2.imread(os.path.join(img_dir, "long_screen.png"), cv2.IMREAD_COLOR)
    my_icon = cv2.imread(os.path.join(chat_dir, name, "my_icon.jpg"), cv2.IMREAD_COLOR)
    other_icon = cv2.imread(os.path.join(chat_dir, name, "other_icon.jpg"), cv2.IMREAD_COLOR)
    my_icon = cv2.resize(my_icon, (108, 108), interpolation=cv2.INTER_AREA)
    other_icon = cv2.resize(other_icon, (108, 108), interpolation=cv2.INTER_AREA)
    chats, msg_data = split_chat_img(long_img, my_icon, other_icon, target_dir=chat_img_dir)

    # 自动更新聊天记录
    start_index = int(img_names[-1].split(".")[0]) + 1
    for i, chat in enumerate(chats):
        cv2.imwrite(os.path.join(chat_img_dir, f"{start_index + i}.png"), chat)

    return


def get_chat_name(img):
    # get the name of the chat from the chat image
    # the chat image should only contain one message
    # return the name of the chat
    assert img.shape[1] == 1080
    y1, x1, y2, x2 = 100, 120, 200, 900
    name_img = img[y1:y2, x1:x2]
    cv2.imwrite(os.path.join(img_dir, "name_img.png"), name_img)
    name = AI.get_text_from_img(name_img)
    return name


def goto_chat(chat_name=None, scroll_limit=5, goto_top=True, attempt_left=3):
    # 该函数在页面找到对应的聊天并点进去
    # 如果chat_name=None，就是找到第一个未读信息
    # 函数返回对应的聊天名称，如果返回None，说明没有未读信息
    # prerequisite: 刚刚已经截过一张图并且已经确认好此页面是chats_page了
    # TODO: 如果两个人的头像一样……
    screen_img = cv2.imread(img_path_map["screen"], cv2.IMREAD_COLOR)
    if chat_name is None:
        # 找到第一个未读信息
        _db_tap(*btn_pos["chats"])
        # 检查是否有未读信息
        title = get_chat_name(screen_img)
        if not ("(" in title or "（" in title):
            # 没有未读信息
            print(f"No unread message: {title}")
            return None
        _tap(*btn_pos["1st_message"])
        screen_img = get_screen()
        return get_chat_name(screen_img)
    # else 指定了聊天对象
    if goto_top:
        # 滚动到画面最顶端
        _db_tap(*btn_pos["weixin"])
    # 通过定位对方的头像来找到对应的聊天
    LEFT, RIGHT = 43, 173  # 头像的左右边界
    TOP, BOTTOM = 220, 2145  # 这是显示聊天的区域
    ICON_SIZE = RIGHT - LEFT
    THRESH = 0.85
    f_name = name2enc.get(chat_name, chat_name)
    f_name = os.path.join(chat_dir, f_name, "other_icon.jpg")
    other_icon = cv2.imread(f_name, cv2.IMREAD_COLOR)
    other_icon = cv2.resize(other_icon, (ICON_SIZE, ICON_SIZE), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(img_dir, "test_other_icon.png"), other_icon)
    icon_mask = cv2.imread(os.path.join(img_dir, "icon_mask.png"), cv2.IMREAD_COLOR)
    screen_img = get_screen()
    temp_img = screen_img[TOP:BOTTOM, LEFT:RIGHT]
    res = cv2.matchTemplate(temp_img, other_icon, cv2.TM_CCOEFF_NORMED, mask=icon_mask)
    if res.max() < THRESH:
        # 没有找到对应的聊天
        if scroll_limit == 0:
            print(f"Error: {chat_name} not found")
            # 重新找一遍
            start_wechat()
            return goto_chat(chat_name, 5, True, attempt_left-1)
        # else: 滚动一个屏幕
        _swipe(540, 2000, 540, 300, 2000)  # 要滚得慢一点，否则可能会超过一个屏幕
        return goto_chat(chat_name, scroll_limit-1, False)
    # else: 找到聊天了
    print(res.max())
    pos = res.argmax().item() + TOP
    _tap(LEFT + ICON_SIZE // 2, pos + ICON_SIZE // 2)
    # 检查对方的名字是否和chat_name一致
    title = get_chat_name(get_screen())
    if title == chat_name:
        return title
    # else: 名字不一致
    print(f"Error: {title} != {chat_name}")
    if attempt_left > 0:
        # 重新找一次
        start_wechat()
        return goto_chat(chat_name, scroll_limit-1, True, attempt_left-1)
    # else: 没有找到
    print(f"Error: {chat_name} not found")
    return None


if __name__ == '__main__':
    # UP_BOUND = 230
    # DOWN_BOUND = 2130
    # img = cv2.imread(os.path.join(img_dir, "screen.png"), cv2.IMREAD_COLOR)[UP_BOUND:DOWN_BOUND]
    # cv2.imwrite(os.path.join(img_dir, "catch_img.png"), img)
    # catch_up_chat(img)
    # pass
    # get_screen()
    # start_wechat()
    # print(goto_chat("妈妈"))
    # start_wechat()
    # catch_up_chat(10)
    # start_wechat()
    # goto_chat("程永康Ken")
    # func_send("你好 你好", "程永康Ken")
    _input_text("哦哦好的")
    # goto_chat("Yongkang CE")
    # print(get_chat_name(get_screen()))
