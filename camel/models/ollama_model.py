# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========
import os
import subprocess
from typing import Any, Dict, Optional, Union

from camel.configs import OLLAMA_API_PARAMS, OllamaConfig
from camel.models.openai_compatible_model import OpenAICompatibleModel
from camel.types import ModelType
from camel.utils import BaseTokenCounter


from json import JSONDecodeError
from typing import Any, Dict, List, Optional, Type, Union

from openai import AsyncOpenAI, AsyncStream, OpenAI, Stream
from pydantic import BaseModel, ValidationError

from camel.logger import get_logger
from camel.messages import OpenAIMessage
from camel.models._utils import try_modify_message_with_format
from camel.models.base_model import BaseModelBackend
from camel.types import (
    ChatCompletion,
    ChatCompletionChunk,
    ModelType,
)
from camel.utils import (
    BaseTokenCounter,
    OpenAITokenCounter,
)

from transformers import AutoTokenizer

class OllamaModel(OpenAICompatibleModel):
    r"""Ollama service interface.

    Args:
        model_type (Union[ModelType, str]): Model for which a backend is
            created.
        model_config_dict (Optional[Dict[str, Any]], optional): A dictionary
            that will be fed into:obj:`openai.ChatCompletion.create()`.
            If:obj:`None`, :obj:`OllamaConfig().as_dict()` will be used.
            (default: :obj:`None`)
        api_key (Optional[str], optional): The API key for authenticating with
            the model service.  Ollama doesn't need API key, it would be
            ignored if set. (default: :obj:`None`)
        url (Optional[str], optional): The url to the model service.
            (default: :obj:`None`)
        token_counter (Optional[BaseTokenCounter], optional): Token counter to
            use for the model. If not provided, :obj:`OpenAITokenCounter(
            ModelType.GPT_4O_MINI)` will be used.
            (default: :obj:`None`)
        timeout (Optional[float], optional): The timeout value in seconds for
            API calls. If not provided, will fall back to the MODEL_TIMEOUT
            environment variable or default to 180 seconds.
            (default: :obj:`None`)

    References:
        https://github.com/ollama/ollama/blob/main/docs/openai.md
    """

    def __init__(
        self,
        model_type: Union[ModelType, str],
        model_config_dict: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
        url: Optional[str] = None,
        token_counter: Optional[BaseTokenCounter] = None,
        timeout: Optional[float] = None,
    ) -> None:
        if model_config_dict is None:
            model_config_dict = OllamaConfig().as_dict()
        url = url or os.environ.get("OLLAMA_BASE_URL")
        timeout = timeout or float(os.environ.get("MODEL_TIMEOUT", 180))
        super().__init__(
            model_type=model_type,
            model_config_dict=model_config_dict,
            api_key=api_key,
            url=url,
            token_counter=token_counter,
            timeout=timeout,
        )

        if not self._url:
            self._start_server()

        tokenizer_path = "D:\workspace\projects\models\megrez2_lenovo"
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    def _start_server(self) -> None:
        r"""Starts the Ollama server in a subprocess."""
        try:
            subprocess.Popen(
                ["ollama", "server", "--port", "11434"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self._url = "http://localhost:11434/v1"
            print(
                f"Ollama server started on {self._url} "
                f"for {self.model_type} model."
            )
        except Exception as e:
            print(f"Failed to start Ollama server: {e}.")

    def check_model_config(self):
        r"""Check whether the model configuration contains any
        unexpected arguments to Ollama API.

        Raises:
            ValueError: If the model configuration dictionary contains any
                unexpected arguments to OpenAI API.
        """
        for param in self.model_config_dict:
            if param not in OLLAMA_API_PARAMS:
                raise ValueError(
                    f"Unexpected argument `{param}` is "
                    "input into Ollama model backend."
                )



    def _request_chat_completion(
        self,
        messages: List[OpenAIMessage],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Union[ChatCompletion, Stream[ChatCompletionChunk]]:
        request_config = self.model_config_dict.copy()

        if tools:
            system_prompt = self.tokenizer.apply_chat_template(messages[:1], add_generation_prompt=False, tokenize=False, tools=tools)
            system_prompt = system_prompt.split("<|role_end|>")[1].split("<|turn_end|>")[0]
            # request_config["tools"] = tools
        else:
            system_prompt = self.tokenizer.apply_chat_template(messages[:1], add_generation_prompt=False, tokenize=False)
        assert messages[0]["role"] == "system"
        messages[0]["content"] = system_prompt

        response = self._client.chat.completions.create(
            messages=messages,
            model=self.model_type,
            **request_config,
        )
        
        # 后处理：解析response中的tool_call内容并构造tool_calls
        if response.choices and response.choices[0].message.content:
            content = response.choices[0].message.content
            tool_calls = self._parse_tool_calls_from_content(content)
            if tool_calls:
                # 构造新的response对象，包含tool_calls
                import copy
                new_response = copy.deepcopy(response)
                new_response.choices[0].message.tool_calls = tool_calls
                return new_response
        
        return response

    async def _arequest_chat_completion(
        self,
        messages: List[OpenAIMessage],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Union[ChatCompletion, AsyncStream[ChatCompletionChunk]]:
        request_config = self.model_config_dict.copy()

        # import pdb; pdb.set_trace()
        if tools:
            system_prompt = self.tokenizer.apply_chat_template(messages[:1], add_generation_prompt=False, tokenize=False, tools=tools)
            system_prompt = system_prompt.split("<|role_end|>")[1].split("<|turn_end|>")[0]
            # request_config["tools"] = tools
        else:
            system_prompt = self.tokenizer.apply_chat_template(messages[:1], add_generation_prompt=False, tokenize=False)
        assert messages[0]["role"] == "system"
        messages[0]["content"] = system_prompt

        response = await self._async_client.chat.completions.create(
            messages=messages,
            model=self.model_type,
            **request_config,
        )
        
        # 后处理：解析response中的tool_call内容并构造tool_calls
        if response.choices and response.choices[0].message.content:
            content = response.choices[0].message.content
            tool_calls = self._parse_tool_calls_from_content(content)
            if tool_calls:
                # 构造新的response对象，包含tool_calls
                import copy
                new_response = copy.deepcopy(response)
                new_response.choices[0].message.tool_calls = tool_calls
                return new_response
        
        return response
    
    def _parse_tool_calls_from_content(self, content: str) -> Optional[List[Any]]:
        """解析content中的tool_call并构造tool_calls列表"""
        import re
        import json
        import uuid
        
        # 匹配<tool_call>标签中的内容
        tool_call_pattern = r'<tool_call>\s*({.*?})\s*</tool_call>'
        matches = re.findall(tool_call_pattern, content, re.DOTALL)
        
        if not matches:
            return None
        
        tool_calls = []
        for i, match in enumerate(matches):
            try:
                # 解析JSON内容
                tool_call_data = json.loads(match)
                tool_name = tool_call_data.get("name")
                arguments = tool_call_data.get("arguments", {})
                
                if tool_name:
                    # 构造tool_call对象，使其具有正确的属性结构
                    class ToolCallObject:
                        def __init__(self, call_id: str, call_type: str, func_name: str, func_args: str):
                            self.id = call_id
                            self.type = call_type
                            self.function = FunctionObject(func_name, func_args)
                    
                    class FunctionObject:
                        def __init__(self, name: str, arguments: str):
                            self.name = name
                            self.arguments = arguments
                    
                    tool_call = ToolCallObject(
                        call_id=f"call_{uuid.uuid4().hex[:8]}",
                        call_type="function",
                        func_name=tool_name,
                        func_args=json.dumps(arguments, ensure_ascii=False)
                    )
                    tool_calls.append(tool_call)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse tool call JSON: {e}")
                continue
        
        return tool_calls if tool_calls else None