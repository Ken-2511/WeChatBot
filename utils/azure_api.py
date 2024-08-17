import requests
from PIL import Image
import io
import json

# 设置 API Key 和 Endpoint
subscription_key = "374f4e67efe4414dbbc0d1d22c0ff4a9"  # 替换为你的 Azure Subscription Key
endpoint = "https://wechatbotocr.cognitiveservices.azure.com/"  # 替换为你的 Azure Endpoint

# 设置 OCR API 的 URL
ocr_url = endpoint + "vision/v3.2/ocr"

# 打开要处理的图像文件
image_path = r"C:\Users\IWMAI\OneDrive\Programs\Python\WeChatBot\utils\images\chats_page.png"
image_data = open(image_path, "rb").read()

# 设置请求头
headers = {
    "Ocp-Apim-Subscription-Key": subscription_key,
    "Content-Type": "application/octet-stream"
}

params = {
    "language": "zh-Hans",
    "detectOrientation": "true"
}

# 发送请求到 Azure OCR API
response = requests.post(ocr_url, headers=headers, data=image_data, params=params)

# 检查响应状态
if response.status_code == 200:
    result = response.json()
    with open("result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

    # 解析并打印 OCR 结果
    for region in result["regions"]:
        for line in region["lines"]:
            line_text = " ".join([word["text"] for word in line["words"]])
            print(line_text)
else:
    print("Error:", response.status_code, response.text)
