import os
import json
from phone_control import *
from message_control import *
from openai import OpenAI
from openai.types.chat import ChatCompletionMessage


class ChatGPT:
    """
    这个类用来管理对话
    """
    def __init__(self, model="gpt-4o-mini"):
        self.client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
        self.model = model
        # 注意这个`self.message`是一个列表，存放了对话记录，但没有包含系统提示
        self.messages = []
        self.intro_message = self._get_intro_message()
        self.tools_message = self._get_tool_message()

    @staticmethod
    def _get_intro_message():
        with open("prompts/intro.txt", "r", encoding="utf-8") as f:
            return {
                "role": "system",
                "content": f.read()
            }

    @staticmethod
    def _get_tool_message():
        with open("prompts/tools.json", "r", encoding="utf-8") as f:
            return json.load(f)

    def request(self):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.get_message(),
            tools=self.tools_message,
            tool_choice="required"
        )
        return response

    def add_message(self, message: dict or ChatCompletionMessage):
        if isinstance(message, dict):
            assert message["role"] in ["user", "system", "tool", "assistant"]
            self.messages.append(message)
        elif isinstance(message, ChatCompletionMessage):
            message = message.to_dict()
            assert message["role"] in ["user", "system", "tool", "assistant"]
            self.messages.append(message)

    def get_message(self):
        self.intro_message = self._get_intro_message()
        self.tools_message = self._get_tool_message()
        messages = [self.intro_message]
        messages.extend(self.messages)
        return messages

    @staticmethod
    def save_chat_completion(response):
        with open("response.json", "w", encoding="utf-8") as f:
            chat_completion_dict = response.to_dict()
            chat_completion_json = json.dumps(chat_completion_dict, indent=4, ensure_ascii=False)
            f.write(chat_completion_json)

    def save_messages(self):
        with open("messages.json", "w", encoding="utf-8") as f:
            messages_json = json.dumps(self.get_message(), indent=4, ensure_ascii=False)
            f.write(messages_json)

    def handle_tool_calls(self, tool_calls) -> list:
        tool_responses = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            if tool_name == "func_send":
                response = func_send(tool_args["content"])
                response["tool_call_id"] = tool_call.id
            elif tool_name == "func_wait":
                response = func_wait(tool_args["n"])
                response["tool_call_id"] = tool_call.id
            else:
                raise ValueError(f"Unknown tool name: {tool_name}")
            tool_responses.append(response)
            self.add_message(response)
        return tool_responses


if __name__ == '__main__':
    chatbot = ChatGPT()
    while True:
        chatbot.add_message({"role": "user", "content": input("User: ")})
        response = chatbot.request()
        chatbot.save_chat_completion(response)
        chatbot.save_messages()
        assert response.choices[0].finish_reason == "tool_call" or response.choices[0].finish_reason == "stop"
        chatbot.add_message(response.choices[0].message)
        tool_responses = chatbot.handle_tool_calls(response.choices[0].message.tool_calls)
