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

import tempfile
from pathlib import Path
from typing import List, Optional
import os

import ffmpeg
from PIL import Image
from scenedetect import (  # type: ignore[import-untyped]
    SceneManager,
    VideoManager,
)
from scenedetect.detectors import (  # type: ignore[import-untyped]
    ContentDetector,
)

from camel.agents import ChatAgent
from camel.configs import QwenConfig
from camel.messages import BaseMessage
from camel.models import ModelFactory, OpenAIAudioModels
from camel.toolkits.base import BaseToolkit
from camel.toolkits.function_tool import FunctionTool
from camel.types import ModelPlatformType, ModelType
from camel.utils import dependencies_required
from loguru import logger

from .video_download_toolkit import (
    VideoDownloaderToolkit,
    _capture_screenshot,
)


class VideoAnalysisToolkit(BaseToolkit):
    r"""Video analysis toolkit using Google's Gemini model.
    
    This toolkit uses the new Google Gen AI SDK (google-genai) instead of 
    the deprecated google-generativeai package.
    
    Installation:
        pip install google-genai
        
    Setup:
        Set your Google API key as an environment variable:
        export GOOGLE_API_KEY="your-api-key-here"
        
    Note:
        - The old google-generativeai package is deprecated as of 2024
        - This toolkit now uses the unified Google Gen AI SDK
        - Supports both local video files and cloud storage URIs
    """
    
    
    def __init__(self, download_directory: Optional[str] = None):
        self.video_downloader_toolkit = VideoDownloaderToolkit(download_directory=download_directory)
    
    
    def ask_question_about_video(self, video_path: str, question: str) -> str:
        r"""Ask a question about the video.
        
        Args:
            video_path (str): The path to the video file.
            question (str): The question to ask about the video.

        Returns:
            str: The answer to the question.
        """
        os.environ["GOOGLE_API_KEY"] = "AIzaSyAAxRMtgD_Zm-clKO6zqMUXnkdqi_NIZm0"
        
        # 使用新的 Google Gen AI SDK
        from google import genai
        from google.genai import types
        
        client = genai.Client()

        # 判断是本地文件还是URI
        if os.path.isfile(video_path):
            # 如果是本地文件，需要先读取并转换为bytes
            with open(video_path, 'rb') as f:
                video_bytes = f.read()
            
            # 根据文件扩展名确定MIME类型
            if video_path.lower().endswith('.mp4'):
                mime_type = 'video/mp4'
            elif video_path.lower().endswith('.avi'):
                mime_type = 'video/avi'
            elif video_path.lower().endswith('.mov'):
                mime_type = 'video/quicktime'
            else:
                mime_type = 'video/mp4'  # 默认类型
            
            video_part = types.Part.from_bytes(data=video_bytes, mime_type=mime_type)
        else:
            # 如果是URI，使用from_uri方法
            video_part = types.Part.from_uri(file_uri=video_path, mime_type='video/mp4')

        response = client.models.generate_content(
            model='gemini-2.0-flash-001',
            contents=[
                types.Part.from_text(text=question),
                video_part
            ]
        )

        logger.debug(f"Video analysis response from gemini: {response.text}")
        return response.text
    
        
    def get_tools(self) -> List[FunctionTool]:
        """
        Get the tools in the toolkit.
        """
        return [FunctionTool(self.ask_question_about_video)]
    

