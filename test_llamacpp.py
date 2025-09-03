#!/usr/bin/env python3
"""
llamacpp客户端测试脚本
用于发送消息到llamacpp server，包括工具调用功能测试
"""

import requests
import json
import time
from typing import Dict, List, Optional, Any


class LlamaCppClient:
    """llamacpp客户端类"""
    
    def __init__(self, server_url: str = "http://localhost:8081"):
        """
        初始化客户端
        
        Args:
            server_url: llamacpp server的URL地址
        """
        self.server_url = server_url.rstrip('/')
        self.completion_url = f"{self.server_url}/completion"
        self.chat_url = f"{self.server_url}/chat/completions"
        
    def test_connection(self) -> bool:
        """
        测试与server的连接
        
        Returns:
            bool: 连接是否成功
        """
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            print(f"连接测试失败: {e}")
            return False
    
    def send_completion_request(self, 
                              prompt: str, 
                              max_tokens: int = 100,
                              temperature: float = 0.7,
                              stop: Optional[List[str]] = None,
                              **kwargs) -> Dict[str, Any]:
        """
        发送completion请求（传统API格式）
        
        Args:
            prompt: 输入提示
            max_tokens: 最大生成token数
            temperature: 温度参数
            stop: 停止词列表
            **kwargs: 其他参数
            
        Returns:
            Dict: 响应结果
        """
        payload = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
            "stop": stop or [],
            **kwargs
        }
        
        try:
            response = requests.post(
                self.completion_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"发送completion请求失败: {e}")
            return {"error": str(e)}
    
    def send_chat_request(self,
                         messages: List[Dict[str, str]],
                         max_tokens: int = 100,
                         temperature: float = 0.7,
                         tools: Optional[List[Dict[str, Any]]] = None,
                         tool_choice: Optional[Dict[str, Any]] = None,
                         **kwargs) -> Dict[str, Any]:
        """
        发送chat completions请求（OpenAI兼容格式）
        
        Args:
            messages: 消息列表，格式为[{"role": "user", "content": "消息内容"}]
            max_tokens: 最大生成token数
            temperature: 温度参数
            tools: 工具列表，用于函数调用
            tool_choice: 工具选择策略
            **kwargs: 其他参数
            
        Returns:
            Dict: 响应结果
        """
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
            **kwargs
        }
        
        # 添加工具调用相关参数
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        
        try:
            response = requests.post(
                self.chat_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"发送chat请求失败: {e}")
            return {"error": str(e)}
    
    def send_tool_call_request(self,
                              messages: List[Dict[str, str]],
                              tools: List[Dict[str, Any]],
                              max_tokens: int = 100,
                              temperature: float = 0.7,
                              tool_choice: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        发送工具调用请求
        
        Args:
            messages: 消息列表
            tools: 工具定义列表
            max_tokens: 最大生成token数
            temperature: 温度参数
            tool_choice: 工具选择策略
            
        Returns:
            Dict: 响应结果
        """
        return self.send_chat_request(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools,
            tool_choice=tool_choice
        )
    
    def simple_chat(self, message: str, max_tokens: int = 100) -> str:
        """
        简单的聊天接口
        
        Args:
            message: 用户消息
            max_tokens: 最大生成token数
            
        Returns:
            str: 模型回复
        """
        messages = [{"role": "user", "content": message}]
        response = self.send_chat_request(messages, max_tokens=max_tokens)
        
        if "error" in response:
            return f"错误: {response['error']}"
        
        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            return f"解析响应失败: {response}"
    
    def extract_tool_calls(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        从响应中提取工具调用
        
        Args:
            response: API响应
            
        Returns:
            List: 工具调用列表
        """
        try:
            message = response["choices"][0]["message"]
            if "tool_calls" in message:
                return message["tool_calls"]
            return []
        except (KeyError, IndexError):
            return []
    
    def create_tool_definition(self, name: str, description: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        创建工具定义
        
        Args:
            name: 工具名称
            description: 工具描述
            parameters: 参数定义
            
        Returns:
            Dict: 工具定义
        """
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters
            }
        }


def create_sample_tools() -> List[Dict[str, Any]]:
    """创建示例工具定义"""
    
    # 计算器工具
    calculator_tool = {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "执行数学计算",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "要计算的数学表达式，如 '2 + 3 * 4'"
                    }
                },
                "required": ["expression"]
            }
        }
    }
    
    # 天气查询工具
    weather_tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称"
                    },
                    "date": {
                        "type": "string",
                        "description": "日期（可选），格式：YYYY-MM-DD"
                    }
                },
                "required": ["city"]
            }
        }
    }
    
    # 翻译工具
    translate_tool = {
        "type": "function",
        "function": {
            "name": "translate",
            "description": "翻译文本",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "要翻译的文本"
                    },
                    "target_language": {
                        "type": "string",
                        "description": "目标语言代码，如 'en', 'zh', 'ja'"
                    },
                    "source_language": {
                        "type": "string",
                        "description": "源语言代码（可选）"
                    }
                },
                "required": ["text", "target_language"]
            }
        }
    }
    
    return [calculator_tool, weather_tool, translate_tool]


def main():
    """主函数 - 演示如何使用客户端"""
    
    # 创建客户端实例
    client = LlamaCppClient("http://localhost:8081")
    
    # 测试连接
    print("正在测试连接...")
    if client.test_connection():
        print("✅ 连接成功!")
    else:
        print("❌ 连接失败，请检查server是否运行")
        return
    
    # # 测试completion API
    # print("\n=== 测试Completion API ===")
    # completion_response = client.send_completion_request(
    #     prompt="你好，请介绍一下你自己",
    #     max_tokens=50,
    #     temperature=0.7
    # )
    # print(f"Completion响应: {json.dumps(completion_response, ensure_ascii=False, indent=2)}")
    
    # # 测试chat API
    # print("\n=== 测试Chat API ===")
    # chat_response = client.send_chat_request(
    #     messages=[{"role": "user", "content": "你好，请介绍一下你自己"}],
    #     max_tokens=50,
    #     temperature=0.7
    # )
    # print(f"Chat响应: {json.dumps(chat_response, ensure_ascii=False, indent=2)}")
    
    # 测试工具调用功能
    print("\n=== 测试工具调用功能 ===")
    
    # 创建示例工具
    tools = create_sample_tools()
    
    # 测试1: 计算器工具调用
    print("\n--- 测试1: 计算器工具调用 ---")
    calculator_test = client.send_tool_call_request(
        messages=[{"role": "user", "content": "请计算 15 + 25 * 2 的结果/no_think"}],
        tools=tools,
        max_tokens=200,
        temperature=0.1
    )
    print(f"计算器工具调用响应: {json.dumps(calculator_test, ensure_ascii=False, indent=2)}")
    
    # 提取工具调用
    tool_calls = client.extract_tool_calls(calculator_test)
    if tool_calls:
        print(f"检测到工具调用: {len(tool_calls)} 个")
        for i, tool_call in enumerate(tool_calls):
            print(f"工具调用 {i+1}:")
            print(f"  函数名: {tool_call.get('function', {}).get('name', 'N/A')}")
            print(f"  参数: {tool_call.get('function', {}).get('arguments', 'N/A')}")
    else:
        print("未检测到工具调用")
    
    # 测试2: 天气查询工具调用
    print("\n--- 测试2: 天气查询工具调用 ---")
    weather_test = client.send_tool_call_request(
        messages=[{"role": "user", "content": "请查询北京的天气情况/no_think"}],
        tools=tools,
        max_tokens=200,
        temperature=0.1
    )
    print(f"天气查询工具调用响应: {json.dumps(weather_test, ensure_ascii=False, indent=2)}")
    
    # 测试3: 翻译工具调用
    print("\n--- 测试3: 翻译工具调用 ---")
    translate_test = client.send_tool_call_request(
        messages=[{"role": "user", "content": "请将 'Hello, how are you?' 翻译成中文/no_think"}],
        tools=tools,
        max_tokens=200,
        temperature=0.1
    )
    print(f"翻译工具调用响应: {json.dumps(translate_test, ensure_ascii=False, indent=2)}")
    
    # 测试4: 强制使用特定工具
    print("\n--- 测试4: 强制使用计算器工具 ---")
    forced_calculator_test = client.send_tool_call_request(
        messages=[{"role": "user", "content": "请帮我计算一下 100 除以 4 的结果/no_think"}],
        tools=tools,
        max_tokens=200,
        temperature=0.1,
        tool_choice={"type": "function", "function": {"name": "calculator"}}
    )
    print(f"强制计算器工具调用响应: {json.dumps(forced_calculator_test, ensure_ascii=False, indent=2)}")
    
    # 测试5: 不使用工具的正常对话
    print("\n--- 测试5: 不使用工具的正常对话 ---")
    normal_chat_test = client.send_tool_call_request(
        messages=[{"role": "user", "content": "请写一首关于春天的诗/no_think"}],
        tools=tools,
        max_tokens=200,
        temperature=0.7,
        tool_choice="none"  # 强制不使用工具
    )
    print(f"正常对话响应: {json.dumps(normal_chat_test, ensure_ascii=False, indent=2)}")
    
    # 测试简单聊天接口
    # print("\n=== 测试简单聊天接口 ===")
    # reply = client.simple_chat("请写一首关于春天的诗", max_tokens=100)
    # print(f"模型回复: {reply}")
    
    # # 交互式聊天
    # print("\n=== 交互式聊天 (输入 'quit' 退出) ===")
    # while True:
    #     user_input = input("\n你: ").strip()
    #     if user_input.lower() in ['quit', 'exit', '退出']:
    #         print("再见!")
    #         break
        
    #     if user_input:
    #         reply = client.simple_chat(user_input, max_tokens=200)
    #         print(f"助手: {reply}")


if __name__ == "__main__":
    main()
