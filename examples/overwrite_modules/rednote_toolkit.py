from camel.toolkits import FunctionTool
from camel.toolkits.base import BaseToolkit
from camel.utils import MCPServer
from playwright.sync_api import sync_playwright
import time
import json
import os
import re
from typing import List


@MCPServer()
class RedNoteToolkit(BaseToolkit):
    r"""A toolkit for reading, creating and managing RedNote.
    """

    def __init__(self):
        self.cookie_file = 'xiaohongshu_cookies.json'
        self.data_dir = 'xiaohongshu_data'
        os.makedirs(self.data_dir, exist_ok=True)

    def _save_cookies(self, context):
        """保存浏览器Cookie到文件"""
        cookies = context.cookies()
        os.makedirs(os.path.dirname(self.cookie_file) or '.', exist_ok=True)
        with open(self.cookie_file, 'w', encoding='utf-8') as f:
            json.dump(cookies, f, ensure_ascii=False, indent=4)
        print(f"Cookie已保存到: {self.cookie_file}")

    def _load_cookies(self, context):
        """从文件加载Cookie到浏览器"""
        if os.path.exists(self.cookie_file):
            with open(self.cookie_file, 'r', encoding='utf-8') as f:
                cookies = json.load(f)
            context.add_cookies(cookies)
            print(f"已从 {self.cookie_file} 加载Cookie")
            return True
        else:
            print(f"Cookie文件 {self.cookie_file} 不存在")
            return False

    def _check_login_status(self, page):
        """检查是否已登录小红书"""
        try:
            avatar = page.locator("div.avatar").first
            if avatar.count() > 0:
                print("已检测到登录状态")
                return True
        except Exception:
            pass

        try:
            login_button = page.locator("text=登录").first
            if login_button.count() > 0:
                print("未检测到登录状态")
                return False
        except Exception:
            pass

        return False

    def _manual_login(self, page, timeout=120):
        """等待用户手动登录并保存Cookie"""
        print(f"请在 {timeout} 秒内完成手动登录")
        page.goto("https://www.xiaohongshu.com/")

        # try:
        #     page.wait_for_selector('span.channel:text("我")', timeout=timeout * 1000)
        #     print("检测到成功登录")
        #     return True
        # except Exception as e:
        #     print(f"登录等待超时或未检测到登录成功: {e}")
        #     return False
        # 每5秒检查一次登录状态
        check_interval = 5
        elapsed_time = 0

        while elapsed_time < timeout:
            print(f"已等待 {elapsed_time} 秒，还剩 {timeout - elapsed_time} 秒...")

            # 检查是否有登录成功的标识
            try:
                # 设置较短的超时时间进行快速检查
                if page.locator('span.channel:text("我")').count() > 0:
                    print("检测到成功登录")
                    return True
            except Exception:
                pass

            # 等待check_interval秒
            time.sleep(check_interval)
            elapsed_time += check_interval

        print(f"超过 {timeout} 秒未检测到登录成功")
        return False

    # def extract_xiaohongshu_notes(self, profile_id: str, force_login: bool = False, max_notes: int = 5):
    def extract_xiaohongshu_notes(self, force_login: bool = False, max_notes: int = 5):
        """
        提取小红书用户的历史笔记内容

        Args:
            profile_id: 小红书用户ID
            force_login: 是否强制重新登录
            max_notes: 最大提取笔记数量

        Returns:
            list: 包含用户笔记内容的列表
        """
        results = []

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)
            context = browser.new_context(viewport={'width': 1280, 'height': 800})
            page = context.new_page()

            login_success = False
            if not force_login:
                if self._load_cookies(context):
                    page.goto("https://www.xiaohongshu.com/")
                    time.sleep(3)
                    login_success = self._check_login_status(page)

            if not login_success:
                print("Cookie登录失败，需要手动登录")
                login_success = self._manual_login(page)
                if login_success:
                    self._save_cookies(context)
                else:
                    print("手动登录失败，继续以未登录状态运行")

            # user_url = f"https://www.xiaohongshu.com/user/profile/{profile_id}"
            # page.goto(user_url)
            # print(f"正在访问用户主页: {user_url}")
            # time.sleep(5)
            print("正在点击'我'按钮跳转到个人主页")
            try:
                # 点击导航栏中的"我"按钮
                me_button = page.locator("span.channel:text('我')").first
                me_button.click()
                print("已点击'我'按钮")
                time.sleep(5)

            except Exception as e:
                print(f"点击'我'按钮或获取profile_id时出错: {e}")
                return []

            # 从当前URL中获取用户ID
            try:
                # 定位"active link-wrapper"元素，这里包含了用户ID
                profile_link = page.locator("a.active.router-link-exact-active.link-wrapper").first
                profile_url = profile_link.get_attribute("href")
                # 从URL中提取用户ID
                profile_id = re.search(r"/user/profile/([^?]+)", profile_url).group(1)
                print(f"获取到用户ID: {profile_id}")
            except Exception as e:
                print(f"获取用户ID时出错: {e}")
                profile_id = "未知ID"

            try:
                note_tab = page.locator("text=笔记").first
                note_tab.click()
                print("已点击笔记标签")
                time.sleep(3)
            except Exception as e:
                print(f"点击笔记标签时出错: {e}")

            note_links = page.locator("a.cover.mask.ld").all()
            if len(note_links) == 0:
                note_links = page.locator("a[class*='cover mask']").all()
                if len(note_links) == 0:
                    note_links = page.locator("section.note-item a[href*='/user/profile/']").all()
                    if len(note_links) == 0:
                        note_links = page.locator("a[href*='/user/profile/']:not([class='name'])").all()

            print(f"找到 {len(note_links)} 个可能的笔记链接")

            max_extract = min(max_notes, len(note_links))

            for i in range(max_extract):
                try:
                    note_link = note_links[i].get_attribute('href')
                    if not note_link.startswith('http'):
                        note_link = 'https://www.xiaohongshu.com' + note_link

                    print(f"正在提取第 {i+1}/{max_extract} 个笔记: {note_link}")

                    note_page = context.new_page()
                    note_page.goto(note_link)
                    time.sleep(3)

                    title_element = note_page.locator("h1").first
                    title = title_element.text_content() if title_element.count() > 0 else "无标题"

                    content = ""
                    try:
                        content_element = note_page.locator("span.note-text").first
                        if content_element.count() > 0:
                            content = content_element.text_content()
                        else:
                            content_element = note_page.locator("div.desc span").first
                            if content_element.count() > 0:
                                content = content_element.text_content()
                            else:
                                content_element = note_page.locator("div#detail-desc").first
                                if content_element.count() > 0:
                                    content = content_element.text_content()
                    except Exception as e:
                        print(f"提取笔记内容时出错: {e}")

                    note_data = {
                        "title": title,
                        "url": note_link,
                        "content": content,
                        "extracted_time": time.strftime("%Y-%m-%d %H:%M:%S")
                    }

                    results.append(note_data)

                    with open(f"{self.data_dir}/{username}_笔记_{i+1}.json", "w", encoding="utf-8") as f:
                        json.dump(note_data, f, ensure_ascii=False, indent=4)

                    note_page.close()

                except Exception as e:
                    print(f"处理笔记时出错: {e}")

            browser.close()

        return results

    def get_tools(self) -> List[FunctionTool]:
        r"""Returns a list of FunctionTool objects representing the
        functions in the toolkit.

        Returns:
            List[FunctionTool]: A list of FunctionTool objects
                representing the functions in the toolkit.
        """
        return [FunctionTool(self.extract_xiaohongshu_notes)]


if __name__ == "__main__":
    # 创建RedNoteToolkit实例
    toolkit = RedNoteToolkit()

    # 测试用户ID，可以替换为任意小红书用户ID
    profile_id = "6809b984000000001b0351a4"

    # 获取小红书用户笔记，设置最大获取3条笔记
    print("开始获取小红书用户笔记...")
    results = toolkit.extract_xiaohongshu_notes(
        # profile_id=profile_id,
        force_login=False,
        max_notes=3
    )

    # 打印获取到的笔记数量
    print(f"成功获取到 {len(results)} 条笔记")

    # 打印每条笔记的标题和部分内容
    for i, note in enumerate(results, 1):
        print(f"\n笔记 {i}:")
        print(f"标题: {note['title']}")
        content_preview = note['content'][:50] + "..." if len(note['content']) > 50 else note['content']
        print(f"内容预览: {content_preview}")
        print(f"链接: {note['url']}")



