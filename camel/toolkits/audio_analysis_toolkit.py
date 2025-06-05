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

import base64
import os
from typing import List, Optional
from urllib.parse import urlparse


import openai
import requests
from pydub.utils import mediainfo

from camel.toolkits.base import BaseToolkit
from camel.toolkits.function_tool import FunctionTool
from camel.agents import ChatAgent
from camel.models import BaseModelBackend

import logging
logger = logging.getLogger(__name__)


class AudioAnalysisToolkit(BaseToolkit):
    r"""A class representing a toolkit for audio operations.

    This class provides methods for processing and understanding audio data.
    """

    def __init__(self, cache_dir: Optional[str] = None, audio_reasoning_model: Optional[BaseModelBackend] = None):
        self.cache_dir = 'tmp/'
        if cache_dir:
            self.cache_dir = cache_dir

        self.client = openai.OpenAI(base_url=os.environ.get("OPENAI_API_BASE_URL"))
        self.audio_reasoning_model = audio_reasoning_model
        
    def get_audio_duration(file_path):
        info = mediainfo(file_path)
        duration = float(info['duration'])
        return duration


    def ask_question_about_audio(self, audio_path: str, question: str) -> str:
        r"""Ask any question about the audio and get the answer using
            multimodal model.

        Args:
            audio_path (str): The path to the audio file.
            question (str): The question to ask about the audio.

        Returns:
            str: The answer to the question.
        """

        logger.debug(
            f"Calling ask_question_about_audio method for audio file \
            `{audio_path}` and question `{question}`."
        )

        # 从环境变量获取API URL，如果没有设置则使用默认值
        api_url = os.environ.get("CHAT_URL", "http://59.110.169.144:7860/chat")
        
        try:
            # 检查音频文件是否存在
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"音频文件不存在: {audio_path}")
            
            # 准备文件参数
            files = []
            with open(audio_path, 'rb') as audio_file:
                # 根据文件扩展名确定MIME类型
                file_ext = os.path.splitext(audio_path)[1].lower()
                if file_ext == '.wav':
                    mime_type = 'audio/wav'
                elif file_ext == '.mp3':
                    mime_type = 'audio/mpeg'
                elif file_ext == '.m4a':
                    mime_type = 'audio/m4a'
                elif file_ext == '.ogg':
                    mime_type = 'audio/ogg'
                else:
                    mime_type = 'audio/wav'  # 默认类型
                
                files.append(('audio_file', (os.path.basename(audio_path), audio_file.read(), mime_type)))
            
            # 准备数据参数
            data = {
                "message": question,
                "return_audio": False
            }
            
            # 发送POST请求
            response = requests.post(api_url, data=data, files=files)
            
            if response.status_code == 200:
                result = response.json()
                return result.get('text', '无法获取回答')
            else:
                error_msg = f"API调用失败: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return error_msg
                
        except FileNotFoundError as e:
            logger.error(f"文件错误: {e}")
            return str(e)
        except Exception as e:
            logger.error(f"处理音频问答时出现错误: {e}")
            return f"处理音频问答时出现错误: {e}"


    def get_tools(self) -> List[FunctionTool]:
        r"""Returns a list of FunctionTool objects representing the functions
            in the toolkit.

        Returns:
            List[FunctionTool]: A list of FunctionTool objects representing the
                functions in the toolkit.
        """
        return [FunctionTool(self.ask_question_about_audio)]
