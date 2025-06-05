from browser_use import Agent
from camel.toolkits.base import BaseToolkit
from typing import Optional
import asyncio
import random
from camel.toolkits import FunctionTool
from playwright.async_api import async_playwright, Browser, BrowserContext, Page
from langchain_openai import ChatOpenAI


OPENAI_API_KEY="sk-c64iuwxrlh3irht6"
OPENAI_API_BASE_URL="https://cloud.infini-ai.com/maas/v1"


class BrowserUseToolkit(BaseToolkit):
    """简化的Browser工具包，每个实例管理自己的browser"""
    
    def __init__(self, headless=True, cache_dir: Optional[str] = None):
        self.headless = headless
        self.cache_dir = cache_dir
        self.playwright = None
        self.browser: Optional[Browser] = None
        self._is_initialized = False

    async def _initialize_browser(self):
        """初始化browser实例"""
        if self._is_initialized:
            return
        
        print("初始化Browser实例...")
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
        print("Browser初始化完成")

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

    async def browse_url(self, task_prompt: str):
        r"""A powerful toolkit which can simulate the browser interaction to solve the task which needs multi-step actions.

        Args:
            task_prompt (str): The task prompt to solve.
        Returns:
            str: The simulation result to the task.
        """
        
        # 创建新的context和page
        context, page = await self._create_context_and_page()
        
        try:
            # 添加随机延迟，模拟人类行为
            await asyncio.sleep(random.uniform(1, 3))
            
            # 创建agent
            agent = Agent(
                task=task_prompt,
                llm=ChatOpenAI(
                    model="gpt-4o-2024-11-20",
                    openai_api_key=OPENAI_API_KEY,
                    openai_api_base=OPENAI_API_BASE_URL
                ),
                browser=self.browser,
                page=page,
            )
            
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
            
            # 执行任务
            state_history = await agent.run()
            return state_history.final_result()
            
        except Exception as e:
            print(f"任务执行失败: {e}")
            raise
        finally:
            # 清理context
            try:
                await context.close()
                print("Context已清理")
            except Exception as e:
                print(f"清理context时出错: {e}")

    async def cleanup(self):
        """清理browser资源"""
        if self.browser:
            try:
                await self.browser.close()
                print("Browser已关闭")
            except Exception as e:
                print(f"关闭browser时出错: {e}")
        
        if self.playwright:
            try:
                await self.playwright.stop()
                print("Playwright已停止")
            except Exception as e:
                print(f"停止playwright时出错: {e}")
        
        self._is_initialized = False

    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self._initialize_browser()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器退出"""
        await self.cleanup()

    def get_tools(self):
        return [FunctionTool(self.browse_url)]


async def test_single_task():
    """测试单个任务"""
    async with BrowserUseToolkit(headless=True, cache_dir="tmp/browser") as toolkit:
        task_prompt = "owl目前有多少stars"
        state_history = await toolkit.browse_url(task_prompt=task_prompt)
        print("任务完成:", state_history)

async def test_multiple_tasks():
    """测试多个任务使用同一个browser"""
    async with BrowserUseToolkit(headless=True, cache_dir="tmp/browser") as toolkit:
        
        tasks = [
            "搜索Python官方文档",
            "搜索最新的AI新闻", 
            "查找Playwright文档"
        ]
        
        for i, task in enumerate(tasks, 1):
            print(f"\n执行任务{i}: {task}")
            result = await toolkit.browse_url(task)
            print(f"任务{i}完成: {result[:100]}...")
    
    print("所有任务完成")

async def test_multiple_browsers():
    """测试多个browser实例并发"""
    tasks = [
        "搜索Python教程",
        "查找JavaScript框架", 
        "搜索机器学习资源"
    ]
    
    async def run_task(task):
        async with BrowserUseToolkit(headless=True, cache_dir="tmp/browser") as toolkit:
            return await toolkit.browse_url(task)
    
    print(f"开始并发执行{len(tasks)}个任务，每个使用独立browser...")
    
    # 并发执行多个任务，每个任务使用独立的browser
    results = await asyncio.gather(*[
        run_task(task) for task in tasks
    ], return_exceptions=True)
    
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"任务{i+1}失败: {result}")
        else:
            print(f"任务{i+1}完成: {str(result)[:100]}...")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        test_type = sys.argv[1]
        if test_type == "single":
            asyncio.run(test_single_task())
        elif test_type == "multiple":
            asyncio.run(test_multiple_tasks())
        elif test_type == "concurrent":
            asyncio.run(test_multiple_browsers())
        else:
            print("用法: python browser_user_toolkit.py [single|multiple|concurrent]")
    else:
        # 默认运行单个任务测试
        asyncio.run(test_single_task())