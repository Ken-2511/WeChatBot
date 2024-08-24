# filename: legacy_code.py
# used to store legacy code that is no longer used but may be useful in the future

import os
import cv2
import numpy as np


def __split_chat_img(long_img, my_icon, other_icon, target_dir=None):
    # 暂时废弃，试着写一个新的更简洁的版本
    # split the long image into several chat images by detecting the icon
    # each chat image should only contain one message. It is either me, or other, or system
    # return the list of chat images

    ICON_SIZE = 108
    MY_X = 940
    OTHER_X = 32
    MESSAGE_X1 = 169
    MESSAGE_X2 = 910
    THRESH = 0.8
    SYS_MSG_H = 153 - 32
    assert my_icon.shape == other_icon.shape == (ICON_SIZE, ICON_SIZE, 3)
    assert long_img.shape[1] == 1080
    h, w, _ = long_img.shape

    # make dir
    if target_dir is None:
        ci_dir = os.path.join(img_dir, "chat_images")  # noqa
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
    time_img = cv2.imread(os.path.join(img_dir, "time_img.png"), cv2.IMREAD_COLOR)  # noqa
    assert time_img.shape == (SYS_MSG_H, 741, 3), time_img.shape
    for i in range(len(icon_pos) - 1):
        assert icon_pos[i] < icon_pos[i + 1] - 32, f"{icon_pos[i]} {icon_pos[i + 1]}"
        chat_img = long_img[icon_pos[i]:icon_pos[i + 1] - 32]
        _h = chat_img.shape[0]
        if chat_img.shape[0] > SYS_MSG_H:
            chat_img = chat_img[chat_img.shape[0] - SYS_MSG_H:, MESSAGE_X1:MESSAGE_X2]
            if _check_difference(chat_img, time_img, threshold=0.1, check_shape=False):  # noqa
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


@staticmethod  # noqa
def __detect_mode(image: np.ndarray):
    ### 废弃。使用detect_mode代替
    # 测试是否是文字
    # 如果是文字的话最好要识别出来
    # 但如果不是文字的话一定要说不是文字
    result = pytesseract.image_to_data(Image.fromarray(image), output_type=pytesseract.Output.DICT,  # noqa
                                       config=AI.config, lang="chi_sim")  # noqa
    if "conf" not in result:
        return "unknown"
    avg_conf = 0
    count = 0
    for conf in result["conf"]:
        if conf == -1:
            continue
        avg_conf += conf
        count += 1
    if count == 0:
        return "unknown"
    avg_conf /= count
    if avg_conf > 85:
        return "text"
    return "unknown"