# WeChatBot
Employ a chatbot on WeChat

让我来说明一下做这个repository的动机：因为最近在淘宝上看到了一些“监督学习”或者是“陪人聊天”的商品。
这些商品好像只要陪人聊天就行了，听起来不难。
我想借此机会创建这样一个repository，使用openai api最大程度模拟人类聊天，能完成图灵测试的那种。

项目分成如下板块：
1. request：负责向openai发送请求
2. message manager：负责管理聊天记录，生成合适的prompt
3. screen_control：负责与手机微信交互

在data文件夹里保存各个聊天记录的文件夹。

data
	chat_id
		screenshots
			imgs
		messages
			imgs
			json
		long_screen_shot