from camel.toolkits import CodeExecutionToolkit
from camel.models import ModelFactory
from dotenv import load_dotenv
from camel.models import ModelFactory
from owl.utils.enhanced_role_playing import OwlRolePlaying, arun_society
from camel.types import ModelPlatformType, ModelType
from examples.overwrite_modules.role_playing import RolePlaying
from camel.agents import ChatAgent
from camel.toolkits import MCPToolkit
import asyncio
import os
import pathlib
from camel.logger import set_log_level

set_log_level(level="DEBUG")


base_dir = pathlib.Path(__file__).parent.parent
env_path = base_dir / ".env"
load_dotenv(dotenv_path=str(env_path))

filepath = os.path.realpath(__file__)
base_dir = os.path.dirname(filepath)


async def get_mcp_toolkit():
    config_path = os.path.join(base_dir, "mcp/lenovo_mcp.json")
    mcp_toolkit = MCPToolkit(config_path=str(config_path))
    await mcp_toolkit.connect()
    return mcp_toolkit


async def aconstruct_society(question: str) -> RolePlaying:
    """
    Construct a society of agents based on the given question.

    Args:
        question (str): The task or question to be addressed by the society.

    Returns:
        RolePlaying: A configured society of agents ready to address the question.
    """
    mcp_toolkit = await get_mcp_toolkit()
    base_url = os.getenv("OPENAI_API_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY")
    model_platform = ModelPlatformType.OPENAI
    model_type_user = ModelType.GPT_4O
    model_type_assistant = ModelType.GPT_4O

    user = ModelFactory.create(
            model_platform=model_platform,
            model_type=model_type_user,
            model_config_dict={"temperature": 0},
            api_key=api_key,
            url=base_url,
        )
    assistant = ModelFactory.create(
        model_platform=model_platform,
        model_type=model_type_assistant,
        model_config_dict={"temperature": 0},
        api_key=api_key,
        url=base_url,
    )

    tools = mcp_toolkit.get_tools()
    print("nr tools: {}".format(len(tools)))
    for tool in tools:
        if hasattr(tool, "get_function_name"):
            print(tool.get_function_name())
            print(tool.get_function_description())
    print("*" * 10)


    planning = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O,
        model_config_dict={"temperature": 0})

    user_profile = """
        用户的照片存放在以下路径:"/Users/zyq/workspace/manus_project/lenovo/data",
        用户的小红书的profile_id: 6809b984000000001b0351a4
    """
    user_agent_kwargs = {"model": user}
    assistant_agent_kwargs = {"model": assistant, "tools": tools}
    society = RolePlaying(
        task_prompt=question,
        with_task_specify=False,
        with_prompt_augmentation=True,
        user_profile=user_profile,
        user_role_name="user",
        user_agent_kwargs=user_agent_kwargs,
        assistant_role_name="assistant",
        assistant_agent_kwargs=assistant_agent_kwargs,
        with_task_planner=True,
        task_planner_agent_kwargs={"model": planning},
        output_language="Chinese")
    print("####")
    return society


def construct_society(question: str) -> RolePlaying:
    return asyncio.run(aconstruct_society(question))


def main():
    default_task = "帮我找一下旅行的风景照，做一个图文并茂的相册，我要发小红书"

    #  default_task = "获取用户小红书历史笔记"
    society = construct_society(default_task)
    answer, chat_history, token_count = asyncio.run(arun_society(society, round_limit=10))
    print(f"\033[94mAnswer: {answer}\033[0m")


if __name__ == "__main__":
    print("start")
    main()
