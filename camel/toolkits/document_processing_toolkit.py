from camel.loaders.chunkr_reader import ChunkrReader
from camel.toolkits.base import BaseToolkit
from camel.toolkits.function_tool import FunctionTool
from camel.toolkits import ImageAnalysisToolkit, AudioAnalysisToolkit, VideoAnalysisToolkit, ExcelToolkit
from camel.messages import BaseMessage
from camel.models import ModelFactory, BaseModelBackend
from camel.types import ModelType, ModelPlatformType
from camel.models import OpenAIModel, DeepSeekModel
from camel.agents import ChatAgent
from docx2markdown._docx_to_markdown import docx_to_markdown
from chunkr_ai import Chunkr
import openai
import requests
import mimetypes
import json
from retry import retry
from typing import List, Dict, Any, Optional, Tuple, Literal, Union
from PIL import Image
from io import BytesIO
from loguru import logger
from bs4 import BeautifulSoup
import asyncio
from urllib.parse import urlparse, urljoin
import os
import subprocess
import xmltodict
from playwright.async_api import Browser, BrowserContext, Page
import asyncio
import nest_asyncio
import random
nest_asyncio.apply()


class DocumentProcessingToolkit(BaseToolkit):
    r"""A class representing a toolkit for processing document and return the content of the document.

    This class provides method for processing docx, pdf, pptx, etc. It cannot process excel files.
    """
    def __init__(self, cache_dir: Optional[str] = None, headless: bool = True):
        self.image_tool = ImageAnalysisToolkit()
        self.audio_tool = AudioAnalysisToolkit()
        self.excel_tool = ExcelToolkit()
        
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        }

        self.cache_dir = "tmp/"
        if cache_dir:
            self.cache_dir = cache_dir
        
        # Browser管理
        self.headless = headless
        self.playwright = None
        self.browser: Optional[Browser] = None
        self._is_initialized = False

    async def _initialize_browser(self):
        """初始化browser实例"""
        if self._is_initialized:
            return
        
        print("初始化DocumentProcessing Browser实例...")
        from playwright.async_api import async_playwright
        self.playwright = await async_playwright().start()
        
        # 反检测启动参数
        launch_args = [
            '--disable-blink-features=AutomationControlled',
            '--disable-dev-shm-usage',
            '--no-sandbox',
            '--disable-setuid-sandbox',
            '--disable-gpu',
            '--disable-web-security',
            '--no-first-run',
            '--disable-background-networking',
            '--disable-background-timer-throttling',
            '--disable-renderer-backgrounding',
            '--disable-backgrounding-occluded-windows',
            '--disable-popup-blocking',
        ]
        
        self.browser = await self.playwright.chromium.launch(
            headless=self.headless,
            args=launch_args
        )
        
        self._is_initialized = True
        print("DocumentProcessing Browser初始化完成")

    async def _create_context_and_page(self) -> tuple[BrowserContext, Page]:
        """创建新的context和page"""
        if not self._is_initialized:
            await self._initialize_browser()
        
        # 创建context
        context = await self.browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            viewport={'width': 1366, 'height': 768},
            locale='zh-CN',
            timezone_id='Asia/Shanghai'
        )
        
        page = await context.new_page()
        
        # 添加反检测脚本
        await page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined,
            });
            
            Object.defineProperty(navigator, 'plugins', {
                get: () => [1, 2, 3, 4, 5],
            });
            
            Object.defineProperty(navigator, 'languages', {
                get: () => ['zh-CN', 'zh', 'en'],
            });
            
            const originalQuery = window.navigator.permissions.query;
            window.navigator.permissions.query = (parameters) => (
                parameters.name === 'notifications' ?
                    Promise.resolve({ state: Notification.permission }) :
                    originalQuery(parameters)
            );
        """)
        
        return context, page

    async def cleanup(self):
        """清理browser资源"""
        if self.browser:
            try:
                await self.browser.close()
                print("DocumentProcessing Browser已关闭")
            except Exception as e:
                print(f"关闭DocumentProcessing browser时出错: {e}")
        
        if self.playwright:
            try:
                await self.playwright.stop()
                print("DocumentProcessing Playwright已停止")
            except Exception as e:
                print(f"停止DocumentProcessing playwright时出错: {e}")
        
        self._is_initialized = False

    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self._initialize_browser()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器退出"""
        await self.cleanup()
    
    @retry((requests.RequestException))
    def extract_document_content(self, document_path: List[str], query: str = None) -> Tuple[bool, str]:
        r"""Extract the content of a list of given documents (or urls) and return the processed text. Try to process the document(s) as much as possible.
        It may filter out some information, resulting in inaccurate content.

        Args:
            document_path (List[str]): The path(s) of the document(s) to be processed, either a local path or a URL. It can process image, audio files, zip files and webpages, etc.
            query (str): The query to be used for retrieving the content. If the content is too long, the query will be used to identify which part contains the relevant information (like RAG). The query should be consistent with the current task.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating whether the document was processed successfully, and the content of the document(s) (if success). When multiple documents are provided, content will be concatenated.
        """
        # Handle list of document paths
        if isinstance(document_path, list):
            logger.debug(f"Calling extract_document_content function with multiple documents: {len(document_path)} files")
            logger.debug(document_path)
            
            # Separate URLs from local files
            urls = []
            local_files = []
            path_index_map = {}  # Maps path to original index
            
            for i, path in enumerate(document_path):
                path_index_map[path] = i
                parsed_url = urlparse(path)
                is_url = all([parsed_url.scheme, parsed_url.netloc])
                if is_url:
                    urls.append(path)
                else:
                    local_files.append(path)
            
            # Process URLs asynchronously in batch
            url_results = {}
            if urls:
                logger.debug(f"Processing {len(urls)} URLs asynchronously")
                url_results = asyncio.run(self._extract_multiple_urls_content(urls, query))
            
            # Process local files synchronously
            local_results = {}
            for path in local_files:
                try:
                    success, content = self._extract_single_document_content(path, query)
                    local_results[path] = (success, content)
                except Exception as e:
                    local_results[path] = (False, str(e))
            
            # Combine results in original order
            all_contents = []
            all_success = True
            
            for i, path in enumerate(document_path):
                if path in url_results:
                    success, content = url_results[path]
                else:
                    success, content = local_results[path]
                
                if success:
                    all_contents.append(f"=== Document {i+1}: {path} ===\n{content}\n")
                else:
                    all_success = False
                    all_contents.append(f"=== Document {i+1}: {path} ===\nFailed to process: {content}\n")
            
            combined_content = "\n".join(all_contents)
            return all_success, combined_content
        
        # Handle single document path (existing logic)
        else:
            logger.debug(f"Calling extract_document_content function with document_path=`{document_path}`")
            return self._extract_single_document_content(document_path, query)
    
    async def _extract_multiple_urls_content(self, urls: List[str], query: str = None) -> Dict[str, Tuple[bool, str]]:
        r"""Extract content from multiple URLs asynchronously.
        
        Args:
            urls (List[str]): List of URLs to process.
            query (str): The query to be used for retrieving the content.
            
        Returns:
            Dict[str, Tuple[bool, str]]: Dictionary mapping URL to (success, content) tuple.
        """
        
        async def _extract_single_url_async(url: str) -> Tuple[str, bool, str]:
            """Extract content from a single URL asynchronously."""
            try:
                # Check if it's a webpage asynchronously
                is_webpage = await self._is_webpage_async(url)
                if is_webpage:
                    extracted_text = await self._extract_webpage_content(url)
                    result_filtered = self._post_process_result(extracted_text, query)
                    return url, True, result_filtered
                else:
                    # For non-webpage URLs, fall back to synchronous processing
                    success, content = self._extract_single_document_content(url, query)
                    return url, success, content
            except Exception as e:
                logger.error(f"Error processing URL {url}: {e}")
                return url, False, str(e)
        
        # Process all URLs concurrently
        tasks = [_extract_single_url_async(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert results to dictionary
        url_results = {}
        for result in results:
            if isinstance(result, Exception):
                # This shouldn't happen with return_exceptions=True, but just in case
                logger.error(f"Unexpected exception in URL processing: {result}")
                continue
            url, success, content = result
            url_results[url] = (success, content)
        
        return url_results

    def _extract_single_document_content(self, document_path: str, query: str = None) -> Tuple[bool, str]:
        r"""Extract the content of a single document (or url) and return the processed text.
        
        Args:
            document_path (str): The path of the document to be processed, either a local path or a URL.
            query (str): The query to be used for retrieving the content.
            
        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating whether the document was processed successfully, and the content of the document (if success).
        """

        if any(document_path.endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
            res = self.image_tool.ask_question_about_image(document_path, "Please make a detailed caption about the image.")
            return True, res
        
        if any(document_path.endswith(ext) for ext in ['.mp3', '.wav']):
            res = self.audio_tool.ask_question_about_audio(document_path, "Please transcribe the audio content to text.")
            return True, res
        
        if any(document_path.endswith(ext) for ext in ['txt']):
            with open(document_path, 'r', encoding='utf-8') as f:
                content = f.read()
            f.close()
            res = self._post_process_result(content, query)
            return True, res
        
        if any(document_path.endswith(ext) for ext in ['xls', 'xlsx']):
            res = self.excel_tool.extract_excel_content(document_path)
            return True, res

        if any(document_path.endswith(ext) for ext in ['zip']): 
            extracted_files = self._unzip_file(document_path)
            return True, f"The extracted files are: {extracted_files}"

        if any(document_path.endswith(ext) for ext in ['json', 'jsonl', 'jsonld']):
            with open(document_path, 'r', encoding='utf-8') as f:
                content = json.load(f)
            f.close()
            return True, content
        
        if any(document_path.endswith(ext) for ext in ['py']):
            with open(document_path, 'r', encoding='utf-8') as f:
                content = f.read()
            f.close()
            return True, content

        
        if any(document_path.endswith(ext) for ext in ['xml']):
            data = None
            with open(document_path, 'r', encoding='utf-8') as f:
                content = f.read()
            f.close()

            try:
                data = xmltodict.parse(content)
                logger.debug(f"The extracted xml data is: {data}")
                return True, data
            
            except Exception as e:
                logger.debug(f"The raw xml data is: {content}")
                return True, content


        if self._is_webpage(document_path):
            extracted_text = asyncio.run(self._extract_webpage_content(document_path))      
            result_filtered = self._post_process_result(extracted_text, query)
            return True, result_filtered
        

        else:
            # judge if url
            parsed_url = urlparse(document_path)
            is_url = all([parsed_url.scheme, parsed_url.netloc])
            if not is_url:
                if not os.path.exists(document_path):
                    return False, f"Document not found at path: {document_path}."

            # if is docx file, use docx2markdown to convert it
            if document_path.endswith(".docx"):
                if is_url:
                    tmp_path = self._download_file(document_path)
                else:
                    tmp_path = document_path
                
                file_name = os.path.basename(tmp_path)
                md_file_path = f"{file_name}.md"
                docx_to_markdown(tmp_path, md_file_path)

                # load content of md file
                with open(md_file_path, "r", encoding="utf-8") as f:
                    extracted_text = f.read()
                f.close()
                return True, extracted_text
            
            if document_path.endswith(".pptx"):
                # use unstructured to extract text from pptx
                try:
                    from unstructured.partition.auto import partition
                    extracted_text = partition(document_path)
                    #return a list of text
                    extracted_text = [item.text for item in extracted_text]
                    return True, extracted_text
                except Exception as e:
                    logger.error(f"Error occurred while processing pptx: {e}")
                    return False, f"Error occurred while processing pptx: {e}"
            
            # try:
            #     result = asyncio.run(self._extract_content_with_chunkr(document_path))
            #     # raise ValueError("Chunkr is not available.")
            #     logger.debug(f"The extracted text from chunkr is: {result}")
            #     result_filtered = self._post_process_result(result, query)
            #     return True, result_filtered

            # except Exception as e:
            #     logger.warning(f"Error occurred while using chunkr to process document: {e}")
            if document_path.endswith(".pdf"):
                # try using pypdf to extract text from pdf
                try:
                    from PyPDF2 import PdfReader
                    if is_url:
                        tmp_path = self._download_file(document_path)
                        document_path = tmp_path

                    with open(document_path, 'rb') as f:
                        reader = PdfReader(f)
                        extracted_text = ""
                        for page in reader.pages:
                            extracted_text += page.extract_text()
                    
                    result_filtered = self._post_process_result(extracted_text, query)
                    return True, result_filtered

                except Exception as e:
                    logger.error(f"Error occurred while processing pdf: {e}")
                    return False, f"Error occurred while processing pdf: {e}"
            
            # use unstructured to extract text from file
            try:
                from unstructured.partition.auto import partition
                extracted_text = partition(document_path)
                #return a list of text
                extracted_text = [item.text for item in extracted_text]
                return True, extracted_text
            
            except Exception as e:
                logger.error(f"Error occurred while processing document: {e}")
                return False, f"Error occurred while processing document: {e}"
    
    
    def _post_process_result(self, result: str, query: str, process_model: BaseModelBackend = None) -> str:
        r"""Identify whether the result is too long. If so, split it into multiple parts, and leverage a model to identify which part contains the relevant information.
        """
        import concurrent.futures
        
        def _identify_relevant_part(part_idx: int, part: str, query: str, _process_model: BaseModelBackend = None) -> Tuple[bool, str]:
            agent = ChatAgent(
                model=_process_model
            )
            
            prompt = f"""
I have retrieved some information from a long document. 
Now I have split the document into multiple parts. Your task is to identify whether the given part contains the relevant information based on the query.

If it does, return only "True". If it doesn't, return only "False". Do not return any other information.

Document part:
<document_part>
{part}
</document_part>

Query:
<query>
{query}
</query>
"""
            
            response = agent.step(prompt)
            if "true" in response.msgs[0].content.lower():
                return True, part_idx, part
            else:
                return False, part_idx, part
        
        
        if process_model is None:
            process_model = ModelFactory.create(
                model_platform=ModelPlatformType.OPENAI,
                # model_type=ModelType.O3_MINI,
                model_type="gpt-4o-2024-11-20",
                model_config_dict={"temperature": 0.0}
            )
            
        max_length = 200000
        split_length = 40000
        
        if len(result) > max_length:
            # split the result into multiple parts
            logger.debug(f"The original result is too long. Splitting it into multiple parts. query: {query}")
            parts = [result[i:i+split_length] for i in range(0, len(result), split_length)]
            result_cache = {}
            # use concurrent.futures to process the parts
            with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
                futures = [executor.submit(_identify_relevant_part, part_idx, part, query, process_model) for part_idx, part in enumerate(parts)]
                for future in concurrent.futures.as_completed(futures):
                    is_relevant, part_idx, part = future.result()
                    if is_relevant:
                        result_cache[part_idx] = part
            # re-assemble the parts according to the part_idx
            result_filtered = ""
            for part_idx in sorted(result_cache.keys()):
                result_filtered += result_cache[part_idx]
                result_filtered += "..."
            
            result_filtered += "(The above is the re-assembled result of the document, because the original document is too long. If empty, it means no relevant information found.)"
            if len(result_filtered) > max_length:
                result_filtered = result_filtered[:max_length]          # TODO: Refine it to be more accurate
            logger.debug(f"split context length: {len(result_filtered)}")
            return result_filtered
        
        else:
            return result


    async def _is_webpage_async(self, url: str) -> bool:
        r"""Judge whether the given URL is a webpage asynchronously."""
        try:
            parsed_url = urlparse(url)
            is_url = all([parsed_url.scheme, parsed_url.netloc])
            if not is_url:
                return False

            path = parsed_url.path
            file_type, _ = mimetypes.guess_type(path)
            if file_type and 'text/html' in file_type:
                return True
            
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.head(url, allow_redirects=True, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    content_type = response.headers.get("Content-Type", "").lower()
                    
                    if "text/html" in content_type:
                        return True
                    else:
                        return False
        
        except Exception as e:
            logger.warning(f"Error while checking the URL: {e}")
            return False

    def _is_webpage(self, url: str) -> bool:
        r"""Judge whether the given URL is a webpage."""
        try:
            parsed_url = urlparse(url)
            is_url = all([parsed_url.scheme, parsed_url.netloc])
            if not is_url:
                return False

            path = parsed_url.path
            file_type, _ = mimetypes.guess_type(path)
            if 'text/html' in file_type:
                return True
            
            response = requests.head(url, allow_redirects=True, timeout=10)
            content_type = response.headers.get("Content-Type", "").lower()
            
            if "text/html" in content_type:
                return True
            else:
                return False
        
        except requests.exceptions.RequestException as e:
            # raise RuntimeError(f"Error while checking the URL: {e}")
            logger.warning(f"Error while checking the URL: {e}")
            return False

        except TypeError:
            return True
    

    @retry(requests.RequestException)
    async def _extract_content_with_chunkr(self, document_path: str, output_format: Literal['json', 'markdown'] = 'markdown') -> str:
        chunkr = Chunkr(api_key=os.getenv("CHUNKR_API_KEY"))
        
        result = await chunkr.upload(document_path)
        
        # result = chunkr.upload(document_path)

        if result.status == "Failed":
            logger.error(f"Error while processing document {document_path}: {result.message}")
            return f"Error while processing document: {result.message}"
        
        # extract document name
        document_name = os.path.basename(document_path)
        output_file_path: str

        if output_format == 'json':
            output_file_path = f"{document_name}.json"
            result.json(output_file_path)

        elif output_format == 'markdown':
            output_file_path = f"{document_name}.md"
            result.markdown(output_file_path)

        else:
            return "Invalid output format."
        
        with open(output_file_path, "r", encoding="utf-8") as f:
            extracted_text = f.read()
        f.close()
        return extracted_text
    
    
    @retry(requests.RequestException, delay=60, backoff=2, max_delay=120)
    async def _extract_webpage_content_with_html2text(self, url: str) -> str:
        import html2text
        import aiohttp
        h = html2text.HTML2Text()
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=self.headers) as response:
                html_content = await response.text()
        
        h.ignore_links = False
        h.ignore_images = False
        h.ignore_tables = False
        extracted_text = h.handle(html_content)
        return extracted_text
    
    @retry(requests.RequestException, delay=60, backoff=2, max_delay=120)
    def _extract_webpage_content_with_beautifulsoup(self, url: str) -> str:
        response = requests.get(url, headers=self.headers)
        html_content = response.text
        soup = BeautifulSoup(html_content, 'html.parser')
        extracted_text = soup.get_text()
        return extracted_text
    

    @retry(RuntimeError, delay=60, backoff=2, max_delay=120)
    async def _extract_webpage_content(self, url: str) -> str:
        """Extract content from a webpage using Playwright."""
        # 创建新的context和page
        context, page = await self._create_context_and_page()
        
        try:
            from playwright.async_api import TimeoutError
            
            # 添加随机延迟，模拟人类行为
            await asyncio.sleep(random.uniform(0.5, 2.0))
            
            # 添加额外的反检测脚本
            await page.evaluate("""
                delete navigator.__proto__.webdriver;
                
                Object.defineProperty(navigator, 'platform', {
                    get: () => 'Win32'
                });
                
                Object.defineProperty(screen, 'availHeight', {
                    get: () => 728
                });
                Object.defineProperty(screen, 'availWidth', {
                    get: () => 1366
                });
            """)
            
            try:
                # Go to the URL and wait only for domcontentloaded
                await page.goto(url, wait_until="domcontentloaded", timeout=15000)
                
                # Wait a short time for any critical dynamic content
                await page.wait_for_timeout(random.randint(1000, 3000))
                
                # Extract text content
                text_content = await page.evaluate("""() => {
                    return document.body.innerText;
                }""")
                
            except TimeoutError:
                logger.warning(f"Timeout while loading {url}, attempting to extract available content")
                # Try to extract whatever content is available
                text_content = await page.evaluate("""() => {
                    return document.body.innerText;
                }""")
            
            if not text_content or len(text_content.strip()) == 0:
                logger.debug("No content found using Playwright, falling back to html2text")
                return await self._extract_webpage_content_with_html2text(url)
            
            return text_content
            
        except Exception as e:
            logger.error(f"Error extracting content with Playwright: {e}")
            logger.debug("Falling back to html2text")
            # If there's a browser error, fallback to html2text
            return await self._extract_webpage_content_with_html2text(url)
        finally:
            # 清理context
            try:
                await context.close()
                print("DocumentProcessing Context已清理")
            except Exception as e:
                print(f"清理DocumentProcessing context时出错: {e}")
    

    def _download_file(self, url: str):
        r"""Download a file from a URL and save it to the cache directory."""
        try:
            response = requests.get(url, stream=True, headers=self.headers)
            response.raise_for_status() 
            file_name = url.split("/")[-1]  

            file_path = os.path.join(self.cache_dir, file_name)

            with open(file_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            
            return file_path

        except requests.exceptions.RequestException as e:
            print(f"Error downloading the file: {e}")


    def _get_formatted_time(self) -> str:
        import time
        return time.strftime("%m%d%H%M")

    
    def _unzip_file(self, zip_path: str) -> List[str]:
        if not zip_path.endswith('.zip'):
            raise ValueError("Only .zip files are supported")
        
        zip_name = os.path.splitext(os.path.basename(zip_path))[0]
        extract_path = os.path.join(self.cache_dir, zip_name)
        os.makedirs(extract_path, exist_ok=True)

        try:
            subprocess.run(["unzip", "-o", zip_path, "-d", extract_path], check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to unzip file: {e}")

        extracted_files = []
        for root, _, files in os.walk(extract_path):
            for file in files:
                extracted_files.append(os.path.join(root, file))
        
        return extracted_files

    def get_tools(self) -> List[FunctionTool]:
        r"""Returns a list of FunctionTool objects representing the functions in the toolkit.

        Returns:
            List[FunctionTool]: A list of FunctionTool objects representing the functions in the toolkit.
        """
        return [
            FunctionTool(self.extract_document_content),
        ]
