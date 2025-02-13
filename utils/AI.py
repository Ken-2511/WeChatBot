import os
import cv2
import base64
import pytesseract
import numpy as np
from enum import Enum
from PIL import Image
from openai import OpenAI
from typing import Literal
from pydantic import BaseModel, constr
from utils.message_classifier import MessageClassifier

platform = "windows"
file_dir = os.path.dirname(__file__)
project_dir = os.path.dirname(file_dir)


class DtypeEnum(str, Enum):
    text = "text"
    image = "image"
    sticker = "sticker"
    voice = "voice"
    file = "file"
    mini_program = "mini_program"
    unknown = "unknown"


class ImgMessage(BaseModel):
    dtype: DtypeEnum
    content: str


class AI:
    if platform == "windows":
        pytesseract.pytesseract.tesseract_cmd = r"C:\Tesseract-OCR\tesseract.exe"
        config = "--tessdata-dir C:/Users/IWMAI/Desktop/tessdata"
    else:
        raise NotImplementedError
    client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
    message_classifier = MessageClassifier(pretrained=True)

    @staticmethod
    def detect_mode(image: np.ndarray):
        return AI.message_classifier.predict_numpy(image)

    @staticmethod
    def detect_unknown(image: np.ndarray):
        # 将复杂的图片交给AI处理，转化为文字描述
        base64_image = base64.b64encode(cv2.imencode(".jpg", image)[1]).decode()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "这是一个微信消息的截图，请你分析信息类型，以及其中的内容。要求描述简短，如果是文字信息，直接输出内容"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "high",
                        }
                    },
                ],
            }
        ]
        response = AI.client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=300,
            response_format=ImgMessage,
        )
        return response.choices[0].message.parsed

    @staticmethod
    def get_text_from_msg_image(image: np.ndarray):
        # 已知第一行文字y坐标35~75，第二行文字y坐标94~134，每行高度为59，文字平均高度为40
        # 可以借此算出第n行文字的y坐标为35+59*(n-1)~75+59*(n-1)
        # 我们根据y坐标中值判断在第几行（从零开始算），公式为 n = ((top+bottom)/2 - 20 - 35 + 29.5) // 59
        # boxes = pytesseract.image_to_boxes(Image.fromarray(image), config=AI.config,
        #                                    output_type=pytesseract.Output.DICT, lang="chi_sim")
        data = pytesseract.image_to_data(Image.fromarray(image), config=AI.config,
                                            output_type=pytesseract.Output.DICT, lang="chi_sim")
        if "text" not in data:
            return "", 999
        img_height = image.shape[0]
        result = []
        for i in range(len(data["text"])):
            text, top, left, width, height = (data["text"][i], data["top"][i],
                                              data["left"][i], data["width"][i], data["height"][i])
            if data["conf"][i] == -1:
                continue
            if data["conf"][i] < 50:
                text = f"_"
            # top = img_height - top
            n = ((top + height / 2) - 20 - 35 + 29.5) // 59
            result.append((text, n, (top, left, width, height)))
        result.sort(key=lambda x: (x[1], x[2][1]))
        if len(result) == 0:
            return "", 999
        ans = ""
        n = 0
        for text, line, _ in result:
            if line > n:
                ans += "\n"
                n = line
            ans += text
        return ans

    @staticmethod
    def describe_image(image: np.ndarray):
        # TODO: describe the image
        return "image"

    @staticmethod
    def get_file_name(image: np.ndarray):
        return AI.get_text_from_msg_image(image)

    @staticmethod
    def __is_chinese(char):
        return '\u4e00' <= char <= '\u9fff'

    @staticmethod
    def __is__english(char):
        return 'a' <= char <= 'z' or 'A' <= char <= 'Z'

    @staticmethod
    def __is_chinese_punctuation(char):
        return char in "，。？！：；、"

    @staticmethod
    def __is_english_punctuation(char):
        return char in ",.?!:;"

    @staticmethod
    def get_text_from_img(image: np.ndarray, remove_newline=True):
        ans = pytesseract.image_to_string(Image.fromarray(image), config=AI.config, lang="chi_sim")
        if remove_newline:
            ans = ans.replace("\n", "")
        # 只有两边都是英文或者左边是英文标点右边是英文才保留空格
        for i in range(len(ans)-2, 0, -1):
            if ans[i] != " ":
                continue
            if AI.__is__english(ans[i-1]) and AI.__is__english(ans[i+1]):
                continue
            if AI.__is_english_punctuation(ans[i-1]) and AI.__is__english(ans[i+1]):
                continue
            ans = ans[:i] + ans[i+1:]
        return ans

    @ staticmethod
    def describe_sticker(image: np.ndarray):
        # TODO: describe the sticker
        return "sticker"


if __name__ == '__main__':
    pass