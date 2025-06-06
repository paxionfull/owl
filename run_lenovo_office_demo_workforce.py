from camel.toolkits import (
    VideoAnalysisToolkit,
    SearchToolkit,
    CodeExecutionToolkit,
    ImageAnalysisToolkit,
    DocumentProcessingToolkit,
    AudioAnalysisToolkit,
    AsyncBrowserToolkit,
    ExcelToolkit,
    FunctionTool
)
from camel.models import ModelFactory
from camel.types import(
    ModelPlatformType,
    ModelType
)
from camel.tasks import Task
from dotenv import load_dotenv

load_dotenv(override=True)

import os
import json
from typing import List, Dict, Any
from loguru import logger
from utils import OwlWorkforceChatAgent, OwlGaiaWorkforce
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from examples.overwrite_modules.email_toolkit import EmailToolkit
from examples.overwrite_modules.office_toolkit import OfficeToolkit


LLM_MODEL = "gpt-4o-2024-11-20"
REASONING_MODEL = "gpt-4o-2024-11-20"


def construct_agent_list() -> List[Dict[str, Any]]:

    web_model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=LLM_MODEL,
        model_config_dict={"temperature": 0},
    )
    
    document_processing_model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=LLM_MODEL,
        model_config_dict={"temperature": 0},
    )
    
    reasoning_model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=REASONING_MODEL,
        model_config_dict={"temperature": 0},
    )
    
    image_analysis_model = ModelFactory.create( 
        model_platform=ModelPlatformType.OPENAI,
        model_type=LLM_MODEL,
        model_config_dict={"temperature": 0},
    )
    
    audio_reasoning_model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=REASONING_MODEL,
        model_config_dict={"temperature": 0},
    )
    
    web_agent_model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=LLM_MODEL,
        model_config_dict={"temperature": 0},
    )
    
    planning_agent_model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=REASONING_MODEL,
        model_config_dict={"temperature": 0},
    )
    

    search_toolkit = SearchToolkit()
    document_processing_toolkit = DocumentProcessingToolkit(cache_dir="tmp")
    image_analysis_toolkit = ImageAnalysisToolkit(model=image_analysis_model)
    video_analysis_toolkit = VideoAnalysisToolkit(download_directory="tmp/video")
    audio_analysis_toolkit = AudioAnalysisToolkit(cache_dir="tmp/audio", audio_reasoning_model=None)
    code_runner_toolkit = CodeExecutionToolkit(sandbox="subprocess", verbose=True)
    # browser_simulator_toolkit = AsyncBrowserToolkit(headless=True, cache_dir=f"tmp/browser", planning_agent_model=planning_agent_model, web_agent_model=web_agent_model)
    excel_toolkit = ExcelToolkit()
    email_toolkit = EmailToolkit()
    office_toolkit = OfficeToolkit()


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
        model=web_model,
        tools=[
            FunctionTool(search_toolkit.search_serper_api),
            FunctionTool(search_toolkit.search_wiki),
            FunctionTool(search_toolkit.search_wiki_revisions),
            FunctionTool(search_toolkit.search_archived_webpage),
            FunctionTool(document_processing_toolkit.extract_document_content),
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
            *office_toolkit.get_tools(),
            *email_toolkit.get_tools(),
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

    agent_list.append(web_agent_dict)
    agent_list.append(document_processing_agent_dict)
    agent_list.append(reasoning_coding_agent_dict)
    return agent_list


def construct_workforce() -> OwlGaiaWorkforce:
    
    coordinator_agent_kwargs = {
        "model": ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=REASONING_MODEL,
            model_config_dict={"temperature": 0},
        )
    }
    
    task_agent_kwargs = {
        "model": ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=LLM_MODEL,
            model_config_dict={"temperature": 0},
        )
    }
    
    answerer_agent_kwargs = {
        "model": ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=LLM_MODEL,
            model_config_dict={"temperature": 0},
        )
    }
    
    workforce = OwlGaiaWorkforce(
        "Lenovo Office Demo Workforce",
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


def process_single_prompt(
    prompt: str,
    file_paths: List[str] = None,
    prompt_id: str = None,
    max_tries: int = 1,
    max_replanning_tries: int = 2,
    thread_id: int = 0
) -> Dict[str, Any]:
    """处理单个自定义prompt"""
    
    # 为每个线程创建独立的workforce
    workforce = construct_workforce()
    
    logger.info(f"Thread {thread_id}: Processing prompt: {prompt[:100]}...")
    
    try:
        # 创建任务
        task = Task(
            content=prompt,
            id=prompt_id or f"custom_task_{thread_id}",
            additional_info=json.dumps({"file_paths": file_paths or []})  # 转换为JSON字符串
        )
        
        # 执行任务
        response = workforce.process_task(
            task=task,
            max_replanning_tries=max_replanning_tries
        )
        
        # 将响应转换为可序列化的格式
        serializable_response = {
            "task_id": response.id,
            "content": response.content,
            "result": response.result,
            "state": str(response.state),
            "additional_info": response.additional_info
        }
            
        return {
            'prompt_id': prompt_id,
            'thread_id': thread_id,
            'prompt': prompt,
            'response': serializable_response,
            'success': True
        }
        
    except Exception as e:
        logger.error(f"Thread {thread_id}: Error processing prompt: {e}")
        return {
            'prompt_id': prompt_id,
            'thread_id': thread_id,
            'prompt': prompt,
            'error': str(e),
            'success': False
        }


def run_demo_prompts():
    """运行联想办公室演示的自定义prompts"""
    
    SAVE_RESULT = True
    MAX_TRIES = 1
    SAVE_RESULT_PATH = "results/lenovo_office_demo/demo_results.json"
    
    # 定义测试prompt
    test_prompt = {
        "id": "market_research",
        "prompt": "请帮我研究一下2024年全球笔记本电脑市场的最新趋势，包括主要品牌的市场份额、技术发展方向以及消费者偏好变化。",
        "files": []
    }

    # 清理临时目录
    if os.path.exists(f"tmp/"):
        shutil.rmtree(f"tmp/")
    
    logger.info(f"Processing prompt: {test_prompt['id']}")
    
    # 处理单个prompt
    result = process_single_prompt(
        prompt=test_prompt["prompt"],
        file_paths=test_prompt.get("files", []),
        prompt_id=test_prompt["id"],
        max_tries=MAX_TRIES,
        max_replanning_tries=2,
        thread_id=0
    )
    
    # 显示结果
    if result['success']:
        logger.success(f"Prompt '{test_prompt['id']}' completed successfully")
        if result.get('response'):
            logger.info(f"Response preview: {str(result['response'])[:200]}...")
    else:
        logger.error(f"Prompt '{test_prompt['id']}' failed: {result.get('error', 'Unknown error')}")
    
    # 保存结果
    if SAVE_RESULT:
        final_result = {
            "total_prompts": 1,
            "successful_prompts": 1 if result['success'] else 0,
            "results": [result]
        }
        
        os.makedirs(os.path.dirname(SAVE_RESULT_PATH), exist_ok=True)
        with open(SAVE_RESULT_PATH, 'w', encoding='utf-8') as f:
            json.dump(final_result, f, indent=2, ensure_ascii=False)
    
    logger.success(f"Processing completed. Success: {result['success']}")


def run_custom_prompt(prompt: str, file_paths: List[str] = None):
    """运行单个自定义prompt的便捷函数"""
    
    logger.info(f"Processing custom prompt: {prompt[:100]}...")
    
    if os.path.exists(f"tmp/"):
        shutil.rmtree(f"tmp/")
    
    result = process_single_prompt(
        prompt=prompt,
        file_paths=file_paths or [],
        prompt_id="custom_single",
        max_tries=1,
        max_replanning_tries=2,
        thread_id=0
    )
    
    if result['success']:
        logger.success("Custom prompt completed successfully")
        logger.info(f"Response: {result['response']}")
        return result['response']
    else:
        logger.error(f"Custom prompt failed: {result.get('error', 'Unknown error')}")
        return None


if __name__ == "__main__":
    # 可以选择运行演示prompts或单个自定义prompt
    
    # 选项1: 运行预定义的演示prompts
    # run_demo_prompts()
    
    # 选项2: 运行单个自定义prompt (取消注释以使用)
    custom_response = run_custom_prompt(
        "查看我电脑上打开的办公文档，并总结我的工作内容, 进而给出第二天详细的工作计划, 用中文回答"
    ) 