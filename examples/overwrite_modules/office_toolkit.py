from io import BytesIO
from typing import List, Optional
from urllib.parse import urlparse

import requests
from PIL import Image

from camel.logger import get_logger
from camel.messages import BaseMessage
from camel.models import BaseModelBackend, ModelFactory
from camel.toolkits import FunctionTool
from camel.toolkits.base import BaseToolkit
from camel.types import ModelPlatformType, ModelType
from camel.utils import MCPServer
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed

import os
from dotenv import load_dotenv

load_dotenv()

logger = get_logger(__name__)


@MCPServer()
class OfficeToolkit(BaseToolkit):
    r"""A toolkit for office tasks.
    """

    def __init__(self, model: Optional[BaseModelBackend] = None):
        r"""Initialize the OfficeToolkit.

        Args:
            model (Optional[BaseModelBackend]): The model backend to use for
                office tasks. If None, a default model will be created using
                ModelFactory. (default: :obj:`None`)
        """
        if model:
            self.model = model
        else:
            self.model = ModelFactory.create(
                model_platform=ModelPlatformType.DEFAULT,
                model_type=ModelType.DEFAULT,
            )

    def image_to_text(
        self, image_path: str, sys_prompt: Optional[str] = None
    ) -> str:
        r"""Generates textual description of an image with optional custom
        prompt.

        Args:
            image_path (str): Local path or URL to an image file.
            sys_prompt (Optional[str]): Custom system prompt for the analysis.
                (default: :obj:`None`)

        Returns:
            str: Natural language description of the image.
        """
        default_content = '''You are an image analysis expert. Provide a
            detailed description including text if present.'''

        system_msg = BaseMessage.make_assistant_message(
            role_name="Senior Computer Vision Analyst, Output in Chinese",
            content=sys_prompt if sys_prompt else default_content,
        )

        return self._analyze_image(
            image_path=image_path,
            prompt="Please describe the contents of this image.",
            system_message=system_msg,
        )

    def images_to_text(
        self, image_path_list: List[str], sys_prompt: Optional[str] = None
    ) -> str:
        r"""Generates textual description of image path list with optional custom prompt.

        Args:
            image_path_list: List[str]: A list of Local path or URL to an image file.
            sys_prompt (Optional[str]): Custom system prompt for the analysis.
                (default: :obj:`None`)

        Returns:
            str: Natural language description of the image.
        """
        result_list = [None] * len(image_path_list)  # 预分配结果列表
        use_threading = True
        max_workers = cpu_count()
        if use_threading:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(
                    self.image_to_text, image_path, sys_prompt): idx
                    for idx, image_path in enumerate(image_path_list)}
                for future in as_completed(futures):
                    idx = futures[future]
                    result_list[idx] = future.result()
        else:
            for idx, image_path in enumerate(image_path_list):
                result_list[idx] = self.image_to_text(image_path, sys_prompt)
        return result_list

    def ask_question_about_image(
        self, image_path: str, question: str, sys_prompt: Optional[str] = None
    ) -> str:
        r"""Answers image questions with optional custom instructions.

        Args:
            image_path (str): Local path or URL to an image file.
            question (str): Query about the image content.
            sys_prompt (Optional[str]): Custom system prompt for the analysis.
                (default: :obj:`None`)

        Returns:
            str: Detailed answer based on visual understanding
        """
        default_content = """Answer questions about images by:
            1. Careful visual inspection
            2. Contextual reasoning
            3. Text transcription where relevant
            4. Logical deduction from visual evidence"""

        system_msg = BaseMessage.make_assistant_message(
            role_name="Visual QA Specialist",
            content=sys_prompt if sys_prompt else default_content,
        )

        return self._analyze_image(
            image_path=image_path,
            prompt=question,
            system_message=system_msg,
        )

    def ask_question_about_images(
        self, image_path_list:List[str], question: str, sys_prompt: Optional[str] = None
    ) -> str:
        r"""Answers image questions with optional custom instructions.

        Args:
            image_path_list (List[str]): Local path or URL to an image file.
            question (str): Query about the image content.
            sys_prompt (Optional[str]): Custom system prompt for the analysis.
                (default: :obj:`None`)

        Returns:
            str: Detailed answer based on visual understanding
        """
        use_threading = True
        max_workers = cpu_count()

        if use_threading:
            result_list = [None] * len(image_path_list)  # 预分配结果列表
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(
                    self.ask_question_about_image, image_path, question, sys_prompt): idx
                    for idx, image_path in enumerate(image_path_list)}
                for future in as_completed(futures):
                    idx = futures[future]
                    result_list[idx] = future.result()
        else:
            result_list = []
            for image_path in image_path_list:
                print("image path: {}".format(image_path))
                result = self.ask_question_about_image(
                    image_path, question, sys_prompt)
                result_list.append(result)
        return result_list

    def _load_image(self, image_path: str) -> Image.Image:
        r"""Loads an image from either local path or URL.

        Args:
            image_path (str): Local path or URL to image.

        Returns:
            Image.Image: Loaded PIL Image object.

        Raises:
            ValueError: For invalid paths/URLs or unreadable images.
            requests.exceptions.RequestException: For URL fetch failures.
        """
        parsed = urlparse(image_path)

        if parsed.scheme in ("http", "https"):
            logger.debug(f"Fetching image from URL: {image_path}")
            try:
                response = requests.get(image_path, timeout=15)
                response.raise_for_status()
                return Image.open(BytesIO(response.content))
            except requests.exceptions.RequestException as e:
                logger.error(f"URL fetch failed: {e}")
                raise
        else:
            logger.debug(f"Loading local image: {image_path}")
            try:
                return Image.open(image_path)
            except Exception as e:
                logger.error(f"Image loading failed: {e}")
                raise ValueError(f"Invalid image file: {e}")

    def _analyze_image(
        self,
        image_path: str,
        prompt: str,
        system_message: BaseMessage,
    ) -> str:
        r"""Core analysis method handling image loading and processing.

        Args:
            image_path (str): Image location.
            prompt (str): Analysis query/instructions.
            system_message (BaseMessage): Custom system prompt for the
                analysis.

        Returns:
            str: Analysis result or error message.
        """
        try:
            image = self._load_image(image_path)
            logger.info(f"Analyzing image: {image_path}")

            from camel.agents.chat_agent import ChatAgent

            agent = ChatAgent(
                system_message=system_message,
                model=self.model,
            )

            user_msg = BaseMessage.make_user_message(
                role_name="User",
                content=prompt,
                image_list=[image],
            )

            response = agent.step(user_msg)
            agent.reset()
            return response.msgs[0].content

        except (ValueError, requests.exceptions.RequestException) as e:
            logger.error(f"Image handling error: {e}")
            return f"Image error: {e!s}"
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return f"Analysis failed: {e!s}"

    def get_tools(self) -> List[FunctionTool]:
        r"""Returns a list of FunctionTool objects representing the functions
            in the toolkit.

        Returns:
            List[FunctionTool]: A list of FunctionTool objects representing the
                functions in the toolkit.
        """
        # FunctionTool(self.ask_question_about_image)

        return [
            FunctionTool(self.images_to_text),
            FunctionTool(self.ask_question_about_images)
        ]



if __name__ == "__main__":
    base_url = os.getenv("OPENAI_API_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY")

    model_platform = ModelPlatformType.QWEN
    model_type = ModelType.QWEN_2_5_VL_72B

    model = ModelFactory.create(
        model_platform=model_platform,
        model_type=model_type,
        model_config_dict={"temperature": 0},
        api_key=api_key,
        url=base_url,
    )
    image_analysis_toolkit = ImageAnalysisToolkit(
        model=model)

    print("#" * 10)
    filepath = "data/IMG_0030.JPG"
    answers = image_analysis_toolkit.ask_question_about_images(
        [filepath, filepath], "图片里面有什么")
    print("#" * 10)
