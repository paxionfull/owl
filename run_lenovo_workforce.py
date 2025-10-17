from camel.toolkits import (
    VideoAnalysisToolkit,
    SearchToolkit,
    CodeExecutionToolkit,
    ImageAnalysisToolkit,
    DocumentProcessingToolkit,
    AudioAnalysisToolkit,
    AsyncBrowserToolkit,
    ExcelToolkit,
    FunctionTool,
)
from camel.models import ModelFactory
from camel.types import(
    ModelPlatformType,
    ModelType
)
from camel.tasks import Task
from dotenv import load_dotenv
from examples.overwrite_modules.email_toolkit import EmailToolkit
from examples.overwrite_modules.office_toolkit import OfficeToolkit

load_dotenv(override=True)

import os
import json
from typing import List, Dict, Any
from loguru import logger
from utils import OwlWorkforceChatAgent, OwlGaiaWorkforce
from utils.gaia import GAIABenchmark
from utils.mint import MINTBenchmark
from utils.hotpotqa import HotpotQABenchmark
from utils.token_tracker import global_token_tracker
from utils.model_decorators import track_token_usage, track_async_token_usage
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
# from examples.overwrite_modules.browser_user_toolkit import BrowserUseToolkit

LLM_MODEL = "gpt-4o-2024-11-20"
REASONING_MODEL = "gpt-4o-2024-11-20"
# LLM_MODEL = "qwen2.5-7b-instruct"
# REASONING_MODEL = "qwen2.5-7b-instruct"
# LLM_MODEL = "/mnt/public/algm/models/Qwen3-4B"
# REASONING_MODEL = "/mnt/public/algm/models/Qwen3-4B"
# LLM_MODEL = "/mnt/public/algm/models/Qwen2.5-4B-Instruct"
# REASONING_MODEL = "/mnt/public/algm/models/Qwen2.5-4B-Instruct"
# LLM_MODEL = "/mnt/public/algm/yzy/train_repos/LLaMA-Factory/saves/qwen3-4b/full/sft/hotpotqa_0_to_300"
# REASONING_MODEL = "/mnt/public/algm/yzy/train_repos/LLaMA-Factory/saves/qwen3-4b/full/sft/hotpotqa_0_to_300"
# LLM_MODEL = "/mnt/public/algm/yzy/train_repos/LLaMA-Factory/saves/qwen2.5-3b/full/sft/hotpotqa_0_to_600"
# REASONING_MODEL = "/mnt/public/algm/yzy/train_repos/LLaMA-Factory/saves/qwen2.5-3b/full/sft/hotpotqa_0_to_600"
# LLM_MODEL = "/mnt/public/algm/yzy/train_repos/LLaMA-Factory/saves/qwen2.5-3b/full/sft/hotpotqa_1_to_600_10e"
# REASONING_MODEL = "/mnt/public/algm/yzy/train_repos/LLaMA-Factory/saves/qwen2.5-3b/full/sft/hotpotqa_1_to_600_10e"
# LLM_MODEL = "/mnt/public/algm/yzy/train_repos/LLaMA-Factory/saves/qwen2.5-3b/full/sft/hotpotqa_1_to_1200_5e"
# REASONING_MODEL = "/mnt/public/algm/yzy/train_repos/LLaMA-Factory/saves/qwen2.5-3b/full/sft/hotpotqa_1_to_1200_5e"
# LLM_MODEL = "/mnt/public/algm/yzy/train_repos/LLaMA-Factory/saves/qwen3-4b/full/sft/hotpotqa_1_to_1200_5e"
# REASONING_MODEL = "/mnt/public/algm/yzy/train_repos/LLaMA-Factory/saves/qwen3-4b/full/sft/hotpotqa_1_to_1200_5e"

WORKER_LLM_MODEL = "gpt-4o-2024-11-20"
WORKER_REASONING_MODEL = "gpt-4o-2024-11-20"
worker_model_platform = ModelPlatformType.OPENAI
worker_url = None
# WORKER_LLM_MODEL = "/mnt/public/algm/yzy/train_repos/LLaMA-Factory/saves/qwen2.5-3b/full/sft/hotpotqa_1_to_1200_10e"
# WORKER_REASONING_MODEL = "/mnt/public/algm/yzy/train_repos/LLaMA-Factory/saves/qwen2.5-3b/full/sft/hotpotqa_1_to_1200_10e"
# WORKER_LLM_MODEL = "/mnt/public/algm/yzy/train_repos/LLaMA-Factory/saves/qwen3-4b/full/sft/hotpotqa_1_to_1200_5e"
# WORKER_REASONING_MODEL = "/mnt/public/algm/yzy/train_repos/LLaMA-Factory/saves/qwen3-4b/full/sft/hotpotqa_1_to_1200_5e"
# WORKER_LLM_MODEL = "/mnt/public/algm/models/Qwen2.5-32B-Instruct"
# WORKER_REASONING_MODEL = "/mnt/public/algm/models/Qwen2.5-32B-Instruct"
# WORKER_LLM_MODEL = "/mnt/public/algm/models/Qwen3-4B-GPTQ-Int4"
# WORKER_REASONING_MODEL = "/mnt/public/algm/models/Qwen3-4B-GPTQ-Int4"
# WORKER_LLM_MODEL = "Qwen3-4B-Q5_K_M"
# WORKER_REASONING_MODEL = "Qwen3-4B-Q5_K_M"
# WORKER_LLM_MODEL = "megrez-moe"
# WORKER_REASONING_MODEL = "megrez-moe"
WORKER_LLM_MODEL = "iter_0000551_Q4_K_M_arc89"
WORKER_REASONING_MODEL = "iter_0000551_Q4_K_M_arc89"
# WORKER_LLM_MODEL = "qwen3-1.7b-gguf"
# WORKER_REASONING_MODEL = "qwen3-1.7b-gguf"
# WORKER_LLM_MODEL = "/mnt/public/algm/yzy/train_repos/LLaMA-Factory/saves/qwen3-4b/full/sft/hotpotqa_1_to_3000_3e"
# WORKER_REASONING_MODEL = "/mnt/public/algm/yzy/train_repos/LLaMA-Factory/saves/qwen3-4b/full/sft/hotpotqa_1_to_3000_3e"
# WORKER_LLM_MODEL = "qwen3-32b"
# WORKER_REASONING_MODEL = "qwen3-32b"
# worker_model_platform = ModelPlatformType.VLLM
worker_model_platform = ModelPlatformType.OLLAMA
worker_url = "http://127.0.0.1:11434/v1"
# worker_url = "http://59.110.169.144:39929/v1"
# worker_model_platform = ModelPlatformType.OPENAI
# worker_url = None

# PIPELINE_LLM_MODEL = "/mnt/public/algm/yzy/models/Qwen2.5-3B-Instruct__21_300_train_jsonl__1-1200__question_v1_1000_decompose_train__8k"
# PIPELINE_REASONING_MODEL = "/mnt/public/algm/yzy/models/Qwen2.5-3B-Instruct__21_300_train_jsonl__1-1200__question_v1_1000_decompose_train__8k"
# PIPELINE_LLM_MODEL = "/mnt/public/algm/yzy/train_repos/LLaMA-Factory/saves/qwen2.5-3b/full/sft/hotpotqa_1_to_1200_10e"
# PIPELINE_REASONING_MODEL = "/mnt/public/algm/yzy/train_repos/LLaMA-Factory/saves/qwen2.5-3b/full/sft/hotpotqa_1_to_1200_10e"
# PIPELINE_LLM_MODEL = "/mnt/public/algm/yzy/train_repos/LLaMA-Factory/saves/qwen3-4b/full/sft/hotpotqa_1_to_1200_5e"
# PIPELINE_REASONING_MODEL = "/mnt/public/algm/yzy/train_repos/LLaMA-Factory/saves/qwen3-4b/full/sft/hotpotqa_1_to_1200_5e"
# PIPELINE_LLM_MODEL = "/mnt/public/algm/models/Qwen2.5-32B-Instruct"
# PIPELINE_REASONING_MODEL = "/mnt/public/algm/models/Qwen2.5-32B-Instruct"
# PIPELINE_LLM_MODEL = "/mnt/public/algm/yzy/train_repos/LLaMA-Factory/saves/qwen3-4b/full/sft/hotpotqa_1_to_3000_3e"
# PIPELINE_REASONING_MODEL = "/mnt/public/algm/yzy/train_repos/LLaMA-Factory/saves/qwen3-4b/full/sft/hotpotqa_1_to_3000_3e"
# PIPELINE_LLM_MODEL = "qwen3-32b"
# PIPELINE_REASONING_MODEL = "qwen3-32b"
# PIPELINE_LLM_MODEL = "/mnt/public/algm/models/Qwen3-4B-GPTQ-Int4"
# PIPELINE_REASONING_MODEL = "/mnt/public/algm/models/Qwen3-4B-GPTQ-Int4"
# PIPELINE_LLM_MODEL = "Qwen3-4B-Q5_K_M"
# PIPELINE_REASONING_MODEL = "Qwen3-4B-Q5_K_M"
# PIPELINE_LLM_MODEL = "megrez-moe"
# PIPELINE_REASONING_MODEL = "megrez-moe"
PIPELINE_LLM_MODEL = "iter_0000551_Q4_K_M_arc89"
PIPELINE_REASONING_MODEL = "iter_0000551_Q4_K_M_arc89"
# PIPELINE_LLM_MODEL = "qwen3-1.7b-gguf"
# PIPELINE_REASONING_MODEL = "qwen3-1.7b-gguf"
# pipeline_model_platform = ModelPlatformType.VLLM
# pipeline_url = "http://59.110.169.14:39929/v1"
pipeline_model_platform = ModelPlatformType.OLLAMA
pipeline_url = "http://127.0.0.1:11434/v1"
# pipeline_model_platform = ModelPlatformType.OPENAI
# pipeline_url = None
# PIPELINE_LLM_MODEL = "gpt-4o-2024-11-20"
# PIPELINE_REASONING_MODEL = "gpt-4o-2024-11-20"
# pipeline_model_platform = ModelPlatformType.OPENAI
# pipeline_url = None


# WORKER_LLM_MODEL = "megrez-moe"
# WORKER_REASONING_MODEL = "megrez-moe"
# PIPELINE_LLM_MODEL = "megrez-moe"
# PIPELINE_REASONING_MODEL = "megrez-moe"
# # worker_model_platform = ModelPlatformType.OPENAI
# worker_model_platform = ModelPlatformType.VLLM
# # worker_url = "https://cloud.infini-ai.com/maas/v1"
# worker_url = "http://localhost:39929/v1"
# # pipeline_model_platform = ModelPlatformType.OPENAI
# pipeline_model_platform = ModelPlatformType.VLLM
# # pipeline_url = "https://cloud.infini-ai.com/maas/v1"
# pipeline_url = "http://localhost:39929/v1"



WORKER_LLM_MODEL = "megrez-moe"
WORKER_REASONING_MODEL = "megrez-moe"
PIPELINE_LLM_MODEL = "megrez-moe"
PIPELINE_REASONING_MODEL = "megrez-moe"
worker_model_platform = ModelPlatformType.OLLAMA
worker_url = "http://127.0.0.1:8081/v1"
pipeline_model_platform = ModelPlatformType.OLLAMA
pipeline_url = "http://127.0.0.1:8081/v1"


# WORKER_LLM_MODEL = "gpt-4o-2024-11-20"
# WORKER_REASONING_MODEL = "gpt-4o-2024-11-20"
# PIPELINE_LLM_MODEL = "gpt-4o-2024-11-20"
# PIPELINE_REASONING_MODEL = "gpt-4o-2024-11-20"
# worker_model_platform = ModelPlatformType.OPENAI
# worker_url = "https://cloud.infini-ai.com/maas/v1"
# pipeline_model_platform = ModelPlatformType.OPENAI
# pipeline_url = "https://cloud.infini-ai.com/maas/v1"


pipeline_model_config_dict = {"temperature": 0}
# worker_model_config_dict = {"temperature": 0}
worker_model_config_dict = {"temperature": 0}
# model_platform = ModelPlatformType.OPENAI
# model_platform = ModelPlatformType.VLLM
# url = None
# # url = "http://59.110.169.144:39929/v1"
# url = "http://127.0.0.1:39929/v1"


def construct_agent_list() -> List[Dict[str, Any]]:

    email_agent_model = ModelFactory.create(
        model_platform=worker_model_platform,
        model_type=WORKER_LLM_MODEL,
        model_config_dict=worker_model_config_dict,
        url=worker_url,
    )
    
    # 为email_agent_model添加token统计
    if hasattr(email_agent_model, '_run'):
        email_agent_model._run = track_token_usage(agent_name="Email Agent")(email_agent_model._run)
    if hasattr(email_agent_model, '_arun'):
        email_agent_model._arun = track_async_token_usage(agent_name="Email Agent")(email_agent_model._arun)

    office_agent_model = ModelFactory.create(
        model_platform=worker_model_platform,
        model_type=WORKER_LLM_MODEL,
        model_config_dict=worker_model_config_dict,
        url=worker_url,
    )
    
    # 为office_agent_model添加token统计
    if hasattr(office_agent_model, '_run'):
        office_agent_model._run = track_token_usage(agent_name="Office Agent")(office_agent_model._run)
    if hasattr(office_agent_model, '_arun'):
        office_agent_model._arun = track_async_token_usage(agent_name="Office Agent")(office_agent_model._arun)

    web_model = ModelFactory.create(
        model_platform=worker_model_platform,
        model_type=WORKER_LLM_MODEL,
        model_config_dict=worker_model_config_dict,
        url=worker_url,
    )
    
    # 为web_model添加token统计
    if hasattr(web_model, '_run'):
        web_model._run = track_token_usage(agent_name="Web Agent")(web_model._run)
    if hasattr(web_model, '_arun'):
        web_model._arun = track_async_token_usage(agent_name="Web Agent")(web_model._arun)
    
    document_processing_model = ModelFactory.create(
        model_platform=worker_model_platform,
        model_type=WORKER_LLM_MODEL,
        model_config_dict=worker_model_config_dict,
        url=worker_url,
    )
    
    # 为document_processing_model添加token统计
    if hasattr(document_processing_model, '_run'):
        document_processing_model._run = track_token_usage(agent_name="Document Processing Agent")(document_processing_model._run)
    if hasattr(document_processing_model, '_arun'):
        document_processing_model._arun = track_async_token_usage(agent_name="Document Processing Agent")(document_processing_model._arun)
    
    reasoning_model = ModelFactory.create(
        model_platform=worker_model_platform,
        model_type=WORKER_REASONING_MODEL,
        model_config_dict=worker_model_config_dict,
        url=worker_url,
    )
    
    # 为reasoning_model添加token统计
    if hasattr(reasoning_model, '_run'):
        reasoning_model._run = track_token_usage(agent_name="Reasoning Coding Agent")(reasoning_model._run)
    if hasattr(reasoning_model, '_arun'):
        reasoning_model._arun = track_async_token_usage(agent_name="Reasoning Coding Agent")(reasoning_model._arun)
    
    image_analysis_model = ModelFactory.create( 
        # model_platform=worker_model_platform,
        # model_type=WORKER_LLM_MODEL,
        # model_config_dict=model_config_dict,
        # url=worker_url,
        model_platform=ModelPlatformType.OPENAI,
        model_type="gpt-4o-2024-11-20",
        model_config_dict=worker_model_config_dict,
        url=None,
    )
    
    # 为image_analysis_model添加token统计
    if hasattr(image_analysis_model, '_run'):
        image_analysis_model._run = track_token_usage(agent_name="Image Analysis Agent")(image_analysis_model._run)
    if hasattr(image_analysis_model, '_arun'):
        image_analysis_model._arun = track_async_token_usage(agent_name="Image Analysis Agent")(image_analysis_model._arun)
    
    audio_reasoning_model = ModelFactory.create(
        model_platform=worker_model_platform,
        model_type=WORKER_REASONING_MODEL,
        model_config_dict=worker_model_config_dict,
        url=worker_url,
    )
    
    # 为audio_reasoning_model添加token统计
    if hasattr(audio_reasoning_model, '_run'):
        audio_reasoning_model._run = track_token_usage(agent_name="Audio Analysis Agent")(audio_reasoning_model._run)
    if hasattr(audio_reasoning_model, '_arun'):
        audio_reasoning_model._arun = track_async_token_usage(agent_name="Audio Analysis Agent")(audio_reasoning_model._arun)
    
    web_agent_model = ModelFactory.create(
        model_platform=worker_model_platform,
        model_type=WORKER_LLM_MODEL,
        model_config_dict=worker_model_config_dict,
        url=worker_url,
    )
    
    # 为web_agent_model添加token统计
    if hasattr(web_agent_model, '_run'):
        web_agent_model._run = track_token_usage(agent_name="Web Agent")(web_agent_model._run)
    if hasattr(web_agent_model, '_arun'):
        web_agent_model._arun = track_async_token_usage(agent_name="Web Agent")(web_agent_model._arun)
    
    planning_agent_model = ModelFactory.create(
        model_platform=worker_model_platform,
        model_type=WORKER_REASONING_MODEL,
        model_config_dict=worker_model_config_dict,
        url=worker_url,
    )
    
    # 为planning_agent_model添加token统计
    if hasattr(planning_agent_model, '_run'):
        planning_agent_model._run = track_token_usage(agent_name="Planning Agent")(planning_agent_model._run)
    if hasattr(planning_agent_model, '_arun'):
        planning_agent_model._arun = track_async_token_usage(agent_name="Planning Agent")(planning_agent_model._arun)
    

    search_toolkit = SearchToolkit()
    document_processing_toolkit = DocumentProcessingToolkit(cache_dir="tmp")
    image_analysis_toolkit = ImageAnalysisToolkit(model=image_analysis_model)
    video_analysis_toolkit = VideoAnalysisToolkit(download_directory="tmp/video")
    # audio_analysis_toolkit = AudioAnalysisToolkit(cache_dir="tmp/audio", audio_reasoning_model=audio_reasoning_model)
    audio_analysis_toolkit = AudioAnalysisToolkit(cache_dir="tmp/audio", audio_reasoning_model=None)
    code_runner_toolkit = CodeExecutionToolkit(sandbox="subprocess", verbose=True)
    # browser_simulator_toolkit = AsyncBrowserToolkit(headless=True, cache_dir=f"tmp/browser", planning_agent_model=planning_agent_model, web_agent_model=web_agent_model)
    excel_toolkit = ExcelToolkit()
    # browser_user_toolkit = BrowserUseToolkit(headless=True)

    email_toolkit = EmailToolkit()
    office_toolkit = OfficeToolkit()

    email_agent = OwlWorkforceChatAgent(
# """
# You are an assistant specialized in analyzing emails and meeting information. You can access Outlook to retrieve email and meeting information.

# tips:
# - If retrieving meeting schedules, please use the get_meetings_on_specific_day tool
# - If the user doesn't clearly specify the time range for emails, please retrieve emails from the last week
# - If retrieving meeting schedules, please return detailed meeting schedule information in the results
# - If analyzing and processing email content, please first return detailed information for each email: title, sender, recipient, sending time, email content. Finally return the analysis conclusion.
# """,
        """
        你是一个专门负责分析邮件和会议信息的助手。你可以通过访问outlook来获取邮件和会议信息。

        tips:
        - 不要重复使用某个工具
        - 如果是获取会议日程，请使用get_meetings_on_specific_day工具
        - 如果用户没有明确表明获取多少时间范围内的邮件，请获取最近一个星期的邮件
        - 如果是获取会议日程，请在结果中返回会议日程的详细信息
        - 如果是分析处理邮件内容，请先在结果中返回各邮件的详细信息：标题，发件人，收件人，发送时间，邮件内容。最后返回分析结论。
        """,
        model=email_agent_model,
        tools=[
            *email_toolkit.get_tools(),
            FunctionTool(code_runner_toolkit.execute_code),  # TODO
        ]
    )

    office_agent = OwlWorkforceChatAgent(
# """
# You are an assistant specialized in analyzing Office document content. You can:
# - Detect all currently open Office documents (Word, Excel, PowerPoint)
# - Extract and analyze Office document content

# Note:
# - Return as complete document content as possible, including absolute file path, title, and document content summary
# - Determine which documents may be related to which to-do items, associate them with the to-do items in the work plan, check how much work remains in the documents, and provide a detailed plan for completing the documents; if the document content is not related to the task, there's no need to include it in the work plan
# - If the user doesn't provide a clear work plan, please provide a possible work plan based on the document content and the user's likely work nature
# - Work plans should not be specific to a certain time point, but rather rough to the level of morning/afternoon
# """,
        """
        你是一个专门负责分析Office文档内容的助手。你可以：
        - 检测当前打开的所有Office文档（Word、Excel、PowerPoint）
        - 提取和分析Office文档内容

        注意：
        - 不要重复使用某个工具
        - 返回文档尽可能完整的内容，包含文件绝对路径，标题，文件内容摘要
        - 判断文档可能与哪个代办事项相关，在工作计划中与该代办事项相关联，查看文档还有多少工作量，并给出完成文档的详细计划；如果文档内容与任务不相关，无需纳入工作计划
        - 如果用户没有提供明确的工作计划，请跟根据文档内容和用户可能的工作性质，给出可能的工作计划
        - 工作计划不要具体到某个时间点，而是粗略到上午下午这种粒度
        """,
        model=office_agent_model,
        tools=[
            *office_toolkit.get_tools(),
        ]
    ) 



    web_agent = OwlWorkforceChatAgent(
"""
You are a helpful assistant that can search the web, extract webpage content, simulate browser actions, and provide relevant information to solve the given task.
Keep in mind that:
- Do not be overly confident in your own knowledge. Searching can provide a broader perspective and help validate existing knowledge.  
- If one way fails to provide an answer, try other ways or methods. The answer does exists.
- If the search snippet is unhelpful but the URL comes from an authoritative source, try visit the website for more details.  
- When looking for specific numerical values (e.g., dollar amounts), prioritize reliable sources and avoid relying only on search snippets.  
- When solving tasks that require web searches, check Wikipedia first before exploring other websites.  
- You can also simulate browser actions to get more information or verify the information you have found.
- If extracting webpage content cannot provide the detailed information about the answer, you should use browser simulation to get more information, else you don't need to use browser simulation.
- Browser simulation is also helpful for finding target URLs. Browser simulation operations do not necessarily need to find specific answers, but can also help find web page URLs that contain answers (usually difficult to find through simple web searches). You can find the answer to the question by performing subsequent operations on the URL, such as extracting the content of the webpage.
- When you are asked question about a video, you don't need to use browser simulation or document tools to find the answer, you should use video analysis toolkit to find the answer.
- Do not solely rely on document tools or browser simulation to find the answer, you should combine document tools and browser simulation to comprehensively process web page information. Some content may need to do browser simulation to get, or some content is rendered by javascript.
- In your response, you should mention the urls you have visited and processed.

Here are some tips that help you perform web search:
- Never add too many keywords in your search query! Some detailed results need to perform browser interaction to get, not using search toolkit.
- If the question is complex, search results typically do not provide precise answers. It is not likely to find the answer directly using search toolkit only, the search query should be concise and focuses on finding official sources rather than direct answers.
  For example, as for the question "What is the maximum length in meters of #9 in the first National Geographic short on YouTube that was ever released according to the Monterey Bay Aquarium website?", your first search term must be coarse-grained like "National Geographic YouTube" to find the youtube website first, and then try other fine-grained search terms step-by-step to find more urls.
- The results you return do not have to directly answer the original question, you only need to collect relevant information.
- If there are multiple documents to be processed, you should process all the documents in the list at once, do not process one by one.
""",
# - If extracting webpage content cannot provide the detailed information about the answer, you should use browser simulation to get more information, else you don't need to use browser simulation.
        model=web_model,
        tools=[
            # FunctionTool(search_toolkit.search_google),
            FunctionTool(search_toolkit.search_serper_api),
            FunctionTool(search_toolkit.search_wiki),
            FunctionTool(search_toolkit.search_wiki_revisions),
            FunctionTool(search_toolkit.search_archived_webpage),
            FunctionTool(document_processing_toolkit.extract_document_content),
            # FunctionTool(browser_simulator_toolkit.browse_url),
            # FunctionTool(browser_user_toolkit.browse_url),
            FunctionTool(video_analysis_toolkit.ask_question_about_video),
        ]
    )
    
    document_processing_agent = OwlWorkforceChatAgent(
        "You are a helpful assistant that can process documents and multimodal data, such as images, audio, and video.",
        document_processing_model,
        tools=[
            FunctionTool(document_processing_toolkit.extract_document_content),
            FunctionTool(image_analysis_toolkit.ask_question_about_image),
            FunctionTool(audio_analysis_toolkit.ask_question_about_audio),
            FunctionTool(video_analysis_toolkit.ask_question_about_video),
            FunctionTool(code_runner_toolkit.execute_code),
        ]
    )
    
    reasoning_coding_agent = OwlWorkforceChatAgent(
        "You are a helpful assistant that specializes in reasoning and coding, and can think step by step to solve the task. When necessary, you can write python code to solve the task. If you have written code, do not forget to execute the code. Never generate codes like 'example code', your code should be able to fully solve the task. You can also leverage multiple libraries, such as requests, BeautifulSoup, re, pandas, etc, to solve the task. For processing excel files, you should write codes to process them.",
        reasoning_model,
        tools=[
            FunctionTool(code_runner_toolkit.execute_code),
            FunctionTool(excel_toolkit.extract_excel_content),
            FunctionTool(document_processing_toolkit.extract_document_content),
        ]
    )

    agent_list = []

    email_agent_dict = {
        "name": "Email Agent",
        "description": "A helpful assistant that can analyze emails and meetings and extract task information",
        "agent": email_agent
    }
    
    office_agent_dict = {
        "name": "Office Agent",
        "description": "A helpful assistant that can analyze office documents and extract relevant information about the task",
        "agent": office_agent
    }
    
    web_agent_dict = {
        "name": "Web Agent",
        "description": "A helpful assistant that can search the web, extract webpage content, simulate browser actions, and retrieve relevant information.",
        "agent": web_agent
    }
    
    document_processing_agent_dict = {
        "name": "Document Processing Agent",
        "description": "A helpful assistant that can process a variety of local and remote documents, including pdf, docx, images, audio, and video, etc.",
        "agent": document_processing_agent
    }
    
    reasoning_coding_agent_dict = {
        "name": "Reasoning Coding Agent",
        "description": "A helpful assistant that specializes in reasoning, coding, and processing excel files. However, it cannot access the internet to search for information. If the task requires python execution, it should be informed to execute the code after writing it.",
        "agent": reasoning_coding_agent
    }

    # agent_list.append(web_agent_dict)
    # agent_list.append(document_processing_agent_dict)
    agent_list.append(reasoning_coding_agent_dict)
    agent_list.append(email_agent_dict)
    agent_list.append(office_agent_dict)
    return agent_list


def construct_workforce() -> OwlGaiaWorkforce:
    
    coordinator_model = ModelFactory.create(
        model_platform=pipeline_model_platform,
        model_type=PIPELINE_REASONING_MODEL,
        model_config_dict=pipeline_model_config_dict,
        url=pipeline_url,
    )
    
    # 为coordinator_model添加token统计
    if hasattr(coordinator_model, '_run'):
        coordinator_model._run = track_token_usage(agent_name="Coordinator Agent")(coordinator_model._run)
    if hasattr(coordinator_model, '_arun'):
        coordinator_model._arun = track_async_token_usage(agent_name="Coordinator Agent")(coordinator_model._arun)
    
    coordinator_agent_kwargs = {
        "model": coordinator_model
    }
    
    task_model = ModelFactory.create(
        model_platform=pipeline_model_platform,
        model_type=PIPELINE_LLM_MODEL,
        model_config_dict=pipeline_model_config_dict,
        url=pipeline_url,
    )
    
    # 为task_model添加token统计
    if hasattr(task_model, '_run'):
        task_model._run = track_token_usage(agent_name="Task Agent")(task_model._run)
    if hasattr(task_model, '_arun'):
        task_model._arun = track_async_token_usage(agent_name="Task Agent")(task_model._arun)
    
    task_agent_kwargs = {
        "model": task_model
    }
    # task_agent_kwargs = {
    #     "model": ModelFactory.create(
    #         model_platform=ModelPlatformType.VLLM,
    #         # model_type="/mnt/public/algm/models/Qwen2.5-3B-Instruct",
    #         # model_type="/mnt/public/algm/models/Qwen3-4B",
    #         # model_type="/mnt/public/algm/zhuangyueqing/public_logs/qwen_sft/Qwen2.5-3B-Instruct__question_v1_1000_decompose_train_jsonl/final",
    #         # model_type="/mnt/public/algm/yzy/models/Qwen2.5-3B-Instruct_question_v1_hermes_data",
    #         # model_type="/mnt/public/algm/yzy/models/qwen3-4b-question_v1_hermes",
    #         # model_type="/mnt/public/algm/yzy/models/qwen3-4b-chat-sft-1106",
    #         # model_type="/mnt/public/algm/yzy/train_repos/LLaMA-Factory/saves/qwen3-4b/full/sft/hotpotqa_0_to_300",
    #         model_type="/mnt/public/algm/yzy/train_repos/LLaMA-Factory/saves/qwen2.5-3b/full/sft/hotpotqa_0_to_300",
    #         model_config_dict=model_config_dict,
    #         # url="http://59.110.169.144:39929/v1",
    #         url="http://127.0.0.1:39929/v1",
    #     )
    # }
    answerer_model = ModelFactory.create(
        model_platform=pipeline_model_platform,
        model_type=PIPELINE_LLM_MODEL,
        model_config_dict=pipeline_model_config_dict,
        url=pipeline_url,
    )
    
    # 为answerer_model添加token统计
    if hasattr(answerer_model, '_run'):
        answerer_model._run = track_token_usage(agent_name="Answerer Agent")(answerer_model._run)
    if hasattr(answerer_model, '_arun'):
        answerer_model._arun = track_async_token_usage(agent_name="Answerer Agent")(answerer_model._arun)
    
    answerer_agent_kwargs = {
        "model": answerer_model
    }
    
    workforce = OwlGaiaWorkforce(
        "Gaia Workforce",
        task_agent_kwargs=task_agent_kwargs,
        coordinator_agent_kwargs=coordinator_agent_kwargs,
        answerer_agent_kwargs=answerer_agent_kwargs
    )

    agent_list = construct_agent_list()
    
    for agent_dict in agent_list:
        workforce.add_single_agent_worker(
            agent_dict["description"],
            worker=agent_dict["agent"],
        )

    return workforce


def process_single_task_index_gaia(
    task_idx: int,
    level: int,
    on: str,
    save_result: bool,
    max_tries: int,
    max_replanning_tries: int,
    data_dir: str,
    result_path: str,
    thread_id: int
) -> Dict[str, Any]:
    """处理单个任务索引，用于并行执行"""
    
    # 重置token统计器
    global_token_tracker.reset()
    global_token_tracker.start_task(f"gaia_task_{task_idx}")
    
    # 为每个线程创建独立的workforce和benchmark
    workforce = construct_workforce()
    
    benchmark = GAIABenchmark(
        data_dir=data_dir,
        save_to=f"{result_path}_thread_{thread_id}.json",
    )
    
    logger.info(f"Thread {thread_id}: Processing task index {task_idx}")
    
    try:
        result = benchmark.run_workforce_with_retry(
            workforce,
            on=on,
            level=level,
            idx=[task_idx],  # 只处理单个任务
            save_result=save_result,
            max_tries=max_tries,
            max_replanning_tries=max_replanning_tries,
        )
        
        # 结束任务统计
        global_token_tracker.end_task(f"gaia_task_{task_idx}")
        
        # 打印token统计报告
        logger.info(f"Thread {thread_id}: Task {task_idx} token统计:")
        global_token_tracker.print_summary()
        
        # 保存详细日志
        if save_result:
            log_path = f"{result_path}_thread_{thread_id}_tokens.json"
            global_token_tracker.save_detailed_log(log_path)
            
        return {
            'task_idx': task_idx,
            'thread_id': thread_id,
            'result': result,
            'token_stats': global_token_tracker.get_summary(),
            'success': True
        }
        
    except Exception as e:
        logger.error(f"Thread {thread_id}: Error processing task {task_idx}: {e}")
        global_token_tracker.end_task(f"gaia_task_{task_idx}")
        return {
            'task_idx': task_idx,
            'thread_id': thread_id,
            'error': str(e),
            'token_stats': global_token_tracker.get_summary(),
            'success': False
        }


def process_single_task_index_mint(
    task_idx: int,
    save_result: bool,
    max_tries: int,
    max_replanning_tries: int,
    data_dir: str,
    result_path: str,
    thread_id: int
) -> Dict[str, Any]:
    """处理单个MINT任务索引，用于并行执行"""
    
    # 为每个线程创建独立的workforce和benchmark
    workforce = construct_workforce()
    
    benchmark = MINTBenchmark(
        data_dir=data_dir,
        save_to=f"{result_path}_thread_{thread_id}.json",
    )
    
    logger.info(f"Thread {thread_id}: Processing MINT task index {task_idx}")
    
    try:
        result = benchmark.run_workforce_with_retry(
            workforce,
            idx=[task_idx],  # 只处理单个任务
            save_result=save_result,
            max_tries=max_tries,
            max_replanning_tries=max_replanning_tries,
        )
            
        return {
            'task_idx': task_idx,
            'thread_id': thread_id,
            'result': result,
            'success': True
        }
        
    except Exception as e:
        logger.error(f"Thread {thread_id}: Error processing MINT task {task_idx}: {e}")
        return {
            'task_idx': task_idx,
            'thread_id': thread_id,
            'error': str(e),
            'success': False
        }


def process_single_task_index_hotpotqa(
    task_idx: int,
    save_result: bool,
    max_tries: int,
    max_replanning_tries: int,
    data_dir: str,
    result_path: str,
    thread_id: int
) -> Dict[str, Any]:
    """处理单个HotpotQA任务索引，用于并行执行"""
    
    # 为每个线程创建独立的workforce和benchmark
    workforce = construct_workforce()
    
    benchmark = HotpotQABenchmark(
        data_dir=data_dir,
        save_to=f"{result_path}_thread_{thread_id}.json",
    )
    
    logger.info(f"Thread {thread_id}: Processing HotpotQA task index {task_idx}")
    
    try:
        result = benchmark.run_workforce_with_retry(
            workforce,
            idx=[task_idx],  # 只处理单个任务
            save_result=save_result,
            max_tries=max_tries,
            max_replanning_tries=max_replanning_tries,
        )
            
        return {
            'task_idx': task_idx,
            'thread_id': thread_id,
            'result': result,
            'success': True
        }
        
    except Exception as e:
        logger.error(f"Thread {thread_id}: Error processing HotpotQA task {task_idx}: {e}")
        return {
            'task_idx': task_idx,
            'thread_id': thread_id,
            'error': str(e),
            'success': False
        }


def check_completed_tasks(result_path: str, task_indices: List[int]) -> List[int]:
    """检查已完成的任务并返回需要处理的任务列表"""
    completed_task_indices = set()
    
    # 检查主结果文件
    if os.path.exists(result_path):
        try:
            with open(result_path, 'r', encoding='utf-8') as f:
                main_result = json.load(f)
                if 'results' in main_result:
                    for result in main_result['results']:
                        # 从结果文件中提取task_id对应的实际任务索引
                        # 这里需要根据实际的数据源来确定索引
                        # 由于task_id是UUID，我们需要检查线程文件来确定对应的索引
                        pass
        except Exception as e:
            logger.warning(f"Error reading main result file: {e}")
    
    # 检查线程特定的结果文件，这些文件的命名包含了任务索引
    base_path = result_path.replace('.json', '')
    for task_idx in task_indices:
        thread_file = f"{base_path}_thread_{task_idx}.json"
        if os.path.exists(thread_file):
            try:
                with open(thread_file, 'r', encoding='utf-8') as f:
                    thread_result = json.load(f)
                    # 检查是否有有效的结果
                    if isinstance(thread_result, list) and len(thread_result) > 0:
                        # 如果文件存在且包含有效结果，认为该任务已完成
                        completed_task_indices.add(task_idx)
                        logger.debug(f"Found completed task file: {thread_file}")
                    elif isinstance(thread_result, dict) and 'results' in thread_result and len(thread_result['results']) > 0:
                        # 如果是字典格式且包含结果，也认为已完成
                        completed_task_indices.add(task_idx)
                        logger.debug(f"Found completed task file: {thread_file}")
            except Exception as e:
                logger.warning(f"Error reading thread result file {thread_file}: {e}")
    
    # 返回未完成的任务列表
    remaining_tasks = [idx for idx in task_indices if idx not in completed_task_indices]
    
    if completed_task_indices:
        logger.info(f"Found {len(completed_task_indices)} completed tasks with indices: {sorted(completed_task_indices)}")
        logger.info(f"Remaining {len(remaining_tasks)} tasks to process: {sorted(remaining_tasks)}")
    else:
        logger.info(f"No completed tasks found, will process all {len(remaining_tasks)} tasks")
    
    return remaining_tasks


def evaluate_on_gaia():
    
    LEVEL = 1
    on="valid"
    SAVE_RESULT = True
    # MAX_TRIES = 3
    MAX_TRIES = 1
    PARALLEL = False  # 新增：是否启用并行处理
    # PARALLEL = True  # 新增：是否启用并行处理
    # MAX_WORKERS = 10  # 新增：最大并行线程数
    MAX_WORKERS = 10  # 新增：最大并行线程数
    
    SAVE_RESULT_PATH = f"results/workforce/workforce_{LEVEL}_pass{MAX_TRIES}_gpt4o.json"
    # test_idx = [16]

    test_idx = [
        30, 
        44
    ] # audio

    test_idx = [
        # 16,
        21
    ] # image
    # test_idx = [
    #     7,
    #     9,
    #     22,
    #     24,
    #     27,
    #     34,
    #     51
    # ] # files

    # video
    # test_idx = [
    #     4,
    #     # 26,
    #     # 33,
    # ] 
    test_idx = [0, 2, 3, 5, 7, 9, 10, 13, 14, 16, 18, 21, 22, 25, 28, 29, 30, 31, 36, 38, 39, 42, 43, 46, 48, 50]

    test_idx = [15]  # browser use
    # test_idx = list(range(53))  # gaia level1
    # test_idx = list(range(43))  # mint hotpotqa

    TASK = "hotpotqa"
    TASK = "gaia"
    TASK = "mint"
    if TASK == "gaia":
        test_idx = list(range(53))
        test_idx = [16]
    elif TASK == "mint":
        # test_idx = list(range(43))
        test_idx = [0]
    elif TASK == "hotpotqa":
        # test_idx = list(range(20))
        # test_idx = list(range(20, 300))
        # test_idx = list(range(300, 600))
        test_idx = list(range(1440, 3000))

    # wrong cases

    if os.path.exists(f"tmp/"):
        shutil.rmtree(f"tmp/")
    
    if PARALLEL and len(test_idx) > 1:
        # 检查已完成的任务，过滤出需要处理的任务
        remaining_tasks = check_completed_tasks(SAVE_RESULT_PATH, test_idx)
        
        if not remaining_tasks:
            logger.info("All tasks have been completed, no need to process")
            return
        
        # 并行处理模式
        logger.info(f"Using parallel processing with {MAX_WORKERS} workers for {len(remaining_tasks)} remaining tasks (originally {len(test_idx)} tasks)")
        
        # 确保结果目录存在
        os.makedirs(os.path.dirname(SAVE_RESULT_PATH), exist_ok=True)
        
        all_results = []
        total_correct = 0
        total_tasks = 0
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # 提交所有任务
            future_to_task = {}
            for i, task_idx in enumerate(remaining_tasks):
                if TASK == "gaia":
                    future = executor.submit(
                        process_single_task_index_gaia,
                        task_idx=task_idx,
                        level=LEVEL,
                        on=on,
                        save_result=SAVE_RESULT,
                        max_tries=MAX_TRIES,
                        max_replanning_tries=2,
                        data_dir="data/gaia",
                        result_path=SAVE_RESULT_PATH.replace('.json', ''),
                        thread_id=task_idx  # 使用任务索引而不是线程序号
                    )
                elif TASK == "mint":
                    future = executor.submit(
                        process_single_task_index_mint,
                        task_idx=task_idx,
                        save_result=SAVE_RESULT,
                        max_tries=MAX_TRIES,
                        max_replanning_tries=2,
                        # data_dir="data/xingyaoww-mint-bench/hotpotqa",
                        data_dir="data/lenovo",
                        result_path=SAVE_RESULT_PATH.replace('.json', ''),
                        thread_id=task_idx  # 使用任务索引而不是线程序号
                    )
                elif TASK == "hotpotqa":
                    future = executor.submit(
                        process_single_task_index_hotpotqa,
                        task_idx=task_idx,
                        save_result=SAVE_RESULT,
                        max_tries=MAX_TRIES,
                        max_replanning_tries=2,
                        data_dir="data/hotpotqa",
                        result_path=SAVE_RESULT_PATH.replace('.json', ''),
                        thread_id=task_idx  # 使用任务索引而不是线程序号
                    )

                future_to_task[future] = task_idx
            
            # 收集结果
            for future in as_completed(future_to_task):
                task_idx = future_to_task[future]
                try:
                    task_result = future.result()
                    if task_result['success']:
                        result = task_result['result']
                        all_results.extend(result.get('results', []))
                        total_correct += result.get('correct', 0)
                        total_tasks += result.get('total', 0)
                        logger.success(f"Task {task_idx} completed. Correct: {result.get('correct', 0)}/{result.get('total', 0)}")
                    else:
                        logger.error(f"Task {task_idx} failed: {task_result.get('error', 'Unknown error')}")
                        total_tasks += 1  # 仍然计入总数
                except Exception as exc:
                    logger.error(f"Task {task_idx} generated an exception: {exc}")
                    total_tasks += 1
        
        # 保存合并后的结果
        if SAVE_RESULT:
            # 如果已有结果文件，先加载现有结果
            existing_results = []
            existing_correct = 0
            existing_total = 0
            
            if os.path.exists(SAVE_RESULT_PATH):
                try:
                    with open(SAVE_RESULT_PATH, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                        existing_results = existing_data.get('results', [])
                        existing_correct = existing_data.get('correct', 0)
                        existing_total = existing_data.get('total', 0)
                except Exception as e:
                    logger.warning(f"Error reading existing results: {e}")
            
            # 合并现有结果和新结果
            combined_results = existing_results + all_results
            combined_correct = existing_correct + total_correct
            combined_total = existing_total + total_tasks
            
            final_result = {
                "total": combined_total,
                "correct": combined_correct,
                "accuracy": combined_correct / combined_total if combined_total > 0 else 0,
                "results": combined_results
            }
            
            with open(SAVE_RESULT_PATH, 'w', encoding='utf-8') as f:
                json.dump(final_result, f, indent=2, ensure_ascii=False)
            
            # 清理线程特定的结果文件
            for task_idx in remaining_tasks:
                thread_file = f"{SAVE_RESULT_PATH.replace('.json', '')}_thread_{task_idx}.json"
                if os.path.exists(thread_file):
                    os.remove(thread_file)
        
        # 获取合并后的统计用于日志输出
        if SAVE_RESULT and os.path.exists(SAVE_RESULT_PATH):
            try:
                with open(SAVE_RESULT_PATH, 'r', encoding='utf-8') as f:
                    final_data = json.load(f)
                    final_correct = final_data.get('correct', total_correct)
                    final_total = final_data.get('total', total_tasks)
                    logger.success(f"Parallel processing completed. New: {total_correct}/{total_tasks}, Combined: {final_correct}/{final_total}")
                    logger.success(f"Combined Accuracy: {final_correct / final_total if final_total > 0 else 0}")
            except Exception as e:
                logger.warning(f"Error reading final results for logging: {e}")
                logger.success(f"Parallel processing completed. Correct: {total_correct}, Total: {total_tasks}")
                logger.success(f"Accuracy: {total_correct / total_tasks if total_tasks > 0 else 0}")
        else:
            logger.success(f"Parallel processing completed. Correct: {total_correct}, Total: {total_tasks}")
            logger.success(f"Accuracy: {total_correct / total_tasks if total_tasks > 0 else 0}")
        
    else:
        # 原始顺序处理模式
        logger.info("Using sequential processing")
        
        # 重置token统计器
        global_token_tracker.reset()
        global_token_tracker.start_task(f"{TASK}_sequential")
        
        if TASK == "gaia":
            benchmark = GAIABenchmark(
                data_dir="data/gaia",
                save_to=SAVE_RESULT_PATH,
            )
            
        elif TASK == "mint":
            benchmark = MINTBenchmark(
                # data_dir="data/xingyaoww-mint-bench/hotpotqa",
                data_dir="data/lenovo",
                save_to=SAVE_RESULT_PATH,
            )
        elif TASK == "hotpotqa":
            benchmark = HotpotQABenchmark(
                data_dir="data/hotpotqa",
                save_to=SAVE_RESULT_PATH,
            )

        workforce = construct_workforce()

        if TASK == "gaia":
            result = benchmark.run_workforce_with_retry(
                workforce,
                on=on,
                level=LEVEL,
                idx=test_idx,
                save_result=SAVE_RESULT,
                max_tries=MAX_TRIES,
                max_replanning_tries=2,
            )
        else:
            result = benchmark.run_workforce_with_retry(
                workforce,
                idx=test_idx,
                save_result=SAVE_RESULT,
                max_tries=MAX_TRIES,
                max_replanning_tries=2,
            )
        
        # 结束任务统计
        global_token_tracker.end_task(f"{TASK}_sequential")
        
        # 打印token统计报告
        logger.info("Token使用统计:")
        global_token_tracker.print_summary()
        
        # 保存详细日志
        if SAVE_RESULT:
            log_path = f"{SAVE_RESULT_PATH.replace('.json', '')}_tokens.json"
            global_token_tracker.save_detailed_log(log_path)
        
        with open(r'.\result.md', 'w', encoding='utf-8') as f:
            f.write(result["results"][0]["model_answer"].strip().replace("```markdown", "").replace("```", "").replace("（2025年6月9日）", "").strip())
        logger.success(f"Correct: {result['correct']}, Total: {result['total']}")
        logger.success(f"Accuracy: {result['accuracy']}")
        
        # 输出token统计到结果文件
        result['token_stats'] = global_token_tracker.get_summary()
        if SAVE_RESULT:
            with open(SAVE_RESULT_PATH, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    evaluate_on_gaia()

