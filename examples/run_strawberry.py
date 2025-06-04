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
import sys
import json
import pathlib
import platform
from dotenv import load_dotenv
from camel.models import ModelFactory
from camel.toolkits import (
    AudioAnalysisToolkit,
    # CodeExecutionToolkit,
    ExcelToolkit,
    # ImageAnalysisToolkit,
    SearchToolkit,
    VideoAnalysisToolkit,
    BrowserToolkit,
    FileWriteToolkit,
)
from camel.types import ModelPlatformType, ModelType, TaskType
from camel.logger import set_log_level
# from camel.societies import RolePlaying
from camel.agents import ChatAgent

from owl.utils import DocumentProcessingToolkit

from examples.overwrite_modules.image_analysis import ImageAnalysisToolkit
from examples.overwrite_modules.role_playing import RolePlaying
from examples.overwrite_modules.rednote_toolkit import RedNoteToolkit
from examples.overwrite_modules.code_execution import CodeExecutionToolkit
from examples.overwrite_modules.agent_prompt import (
    planner_agent_prompt,
    reflection_agent_prompt,
    history_summary_agent_prompt
)
from owl.utils.strawberry import run_society

base_dir = pathlib.Path(__file__).parent.parent
env_path = base_dir / "owl" / ".env"
load_dotenv(dotenv_path=str(env_path))

set_log_level(level="DEBUG")


def construct_society(question: str) -> RolePlaying:
    r"""Construct a society of agents based on the given question.

    Args:
        question (str): The task or question to be addressed by the society.

    Returns:
        RolePlaying: A configured society of agents ready to address the question.
    """

    LLM_MODEL = "gpt-4o-2024-11-20"

    # Create models for different components
    models = {
        "user": ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=LLM_MODEL,
            model_config_dict={"temperature": 0},
        ),
        "assistant": ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=LLM_MODEL,
            model_config_dict={"temperature": 0},
        ),
        "browsing": ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=LLM_MODEL,
            model_config_dict={"temperature": 0},
        ),
        "planning": ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=LLM_MODEL,
            model_config_dict={"temperature": 0},
        ),
        "video": ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=LLM_MODEL,
            model_config_dict={"temperature": 0},
        ),
        "image": ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=LLM_MODEL,
            model_config_dict={"temperature": 0},
        ),
        "document": ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=LLM_MODEL,
            model_config_dict={"temperature": 0},
        ),
        "task_specify": ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=LLM_MODEL,
            model_config_dict={"temperature": 0},
        ),
        "history_summary": ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=LLM_MODEL,
            model_config_dict={"temperature": 0},
        ),
        "reflection": ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=LLM_MODEL,
            model_config_dict={"temperature": 0},
        ),
    }

    # Configure toolkits
    tools = [
        # *BrowserToolkit(
        #     headless=False,  # Set to True for headless mode (e.g., on remote servers)
        #     web_agent_model=models["browsing"],
        #     planning_agent_model=models["planning"],
        # ).get_tools(),
        # *VideoAnalysisToolkit(model=models["video"]).get_tools(),
        # *AudioAnalysisToolkit().get_tools(),  # This requires OpenAI Key
        *CodeExecutionToolkit(sandbox="subprocess", verbose=True).get_tools(),
        *ImageAnalysisToolkit(model=models["image"]).get_tools(),
        # SearchToolkit().search_duckduckgo,
        # SearchToolkit().search_google,  # Comment this out if you don't have google search
        # SearchToolkit().search_wiki,
        # *ExcelToolkit().get_tools(),
        *DocumentProcessingToolkit(model=models["document"]).get_tools(),
        *FileWriteToolkit(output_dir="./").get_tools(),
        *RedNoteToolkit().get_tools(),
    ]

    # 根据操作系统类型设置不同的路径
    system = platform.system()
    if system == "Windows":
        user_profile = """用户的照片存放在以下路径:
    "C:\\Users\\Device Intelligent\\Pictures\\American"
"""
    else:  # macOS或其他Unix系统
        user_profile = """用户的照片存放在以下路径:
    "/Users/zhuyuyao/Documents/llm应用/联想demo/美国旅行"
"""

    # Configure agent roles and parameters
    user_agent_kwargs = {"model": models["user"]}
    assistant_agent_kwargs = {"model": models["assistant"], "tools": tools}
    task_specify_agent_kwargs = {"model": models["task_specify"]}
    task_planner_agent_kwargs = {"model": models["planning"]}
    prompt_augmentation_agent_kwargs = {"model": models["task_specify"]}
    history_summary_agent_kwargs = {"model": models["history_summary"]}
    reflection_agent_kwargs = {"model": models["reflection"]}

    # Configure task parameters
    task_kwargs = {
        # "task_prompt": question,
        "with_task_specify": False,
        # "task_type": TaskType.AI_SOCIETY,
        # "with_task_specify": True,
        # "task_specify_agent_kwargs": task_specify_agent_kwargs,
        # "with_task_planner": True,
        # "task_planner_agent_kwargs": task_planner_agent_kwargs,
        "user_profile": user_profile,
        # "with_prompt_augmentation": True,
        "with_prompt_augmentation": False,
        "prompt_augmentation_agent_kwargs": prompt_augmentation_agent_kwargs,
    }

    # Create and return the society
    society = RolePlaying(
        **task_kwargs,
        user_role_name="user",
        user_agent_kwargs=user_agent_kwargs,
        assistant_role_name="assistant",
        assistant_agent_kwargs=assistant_agent_kwargs,
    )

    guideline_prompt = """优化prompt遵循如下规则:
- 如果task_prompt中欠缺完成任务的必要信息，例如文件路径。请查看user_profile中是否提供了相关信息, 并相应添加在优化后的task prompt中。
- 如果task_prompt中出现了对文案风格的要求，比如说"发小红书"，"发微博"之类的。优化后的prompt要明确保留相应的风格要求，写为例如"使用小红书的文案风格"。
- 最后的文案默认写入文件outputs/tmp.html. 
"""

    planner_agent = ChatAgent(
        planner_agent_prompt.format(
            tools="\n".join([json.dumps(tool.get_openai_tool_schema(), ensure_ascii=False) for tool in tools]),
            user_profile=society.user_profile,
            guideline=society.guideline_prompt,
        ),
        output_language="Chinese",
        **(task_planner_agent_kwargs or {}),
    )
    history_summary_agent = ChatAgent(
        history_summary_agent_prompt,
        output_language="Chinese",
        **(history_summary_agent_kwargs or {}),
    )

    reflection_agent = ChatAgent(
        reflection_agent_prompt,
        output_language="Chinese",
        **(assistant_agent_kwargs or {}),
    )

    return {
        "society": society,
        "planner_agent": planner_agent,
        "reflection_agent": reflection_agent,
        "history_summary_agent": history_summary_agent,
        # task
        "task_prompt": question,
        "task_kwargs": task_kwargs,
        "user_agent_kwargs": user_agent_kwargs,
        "assistant_agent_kwargs": assistant_agent_kwargs,
        "guideline_prompt": guideline_prompt,
    }


# def main():
#     r"""Main function to run the OWL system with an example question."""
#     # Default research question
#     # default_task = "Navigate to Amazon.com and identify one product that is attractive to coders. Please provide me with the product name and price. No need to verify your answer."
#     # default_task = "搜索大疆action 5pro的价格"
#     default_task = (
#         "查看目录\"/Users/zhuyuyao/Documents/llm应用/联想demo/旅行\"下的照片，根据图片内容生成一个旅游游记markdown, 存储在当前目录的trip_log_mcp.md中。markdown内容包含图片(图片引用使用绝对路径)以及对应的内容描述，整体构成一个完整旅行叙事。"
#     )

#     # Override default task if command line argument is provided
#     task = sys.argv[1] if len(sys.argv) > 1 else default_task

#     # Construct and run the society
#     society = construct_society(task)
#     answer, chat_history, token_count = run_society(society)

#     # Output the result
#     print(f"\033[94mAnswer: {answer}\033[0m")


def main():
    r"""Main function to run the OWL system with an example question."""
    # Default research question
    # default_task = "Navigate to Amazon.com and identify one product that is attractive to coders. Please provide me with the product name and price. No need to verify your answer."
    # default_task = "搜索大疆action 5pro的价格"
    # default_task = (
    #     "查看目录\"/Users/zhuyuyao/Documents/llm应用/联想demo/旅行\"下的照片，根据图片内容生成一个旅游游记markdown, 存储在当前目录的trip_log_mcp.md中。markdown内容包含图片(图片引用使用绝对路径)以及对应的内容描述，整体构成一个完整旅行叙事。"
    # )

    default_task = "帮我找一下旅行的风景照，做一个图文并茂的相册，我要发小红书"

    # Override default task if command line argument is provided
    task = sys.argv[1] if len(sys.argv) > 1 else default_task

    # Construct and run the society
    society = construct_society(task)
    answer, chat_history, token_count = run_society(society, round_limit=5)

    # Output the result
    print(f"\033[94mAnswer: {answer}\033[0m")


if __name__ == "__main__":
    main()