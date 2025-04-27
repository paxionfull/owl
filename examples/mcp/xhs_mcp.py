from playwright.sync_api import sync_playwright
from playwright.async_api import async_playwright
import time
import json
import os
import re
from fastmcp import FastMCP
import asyncio
server = FastMCP("xhs-tools")
cookie_file="/Users/zyq/xiaohongshu_cookies.json"

async def save_cookies(context, cookie_file="xiaohongshu_cookies.json"):
    """保存浏览器Cookie到文件"""
    cookies = await context.cookies()
    os.makedirs(os.path.dirname(cookie_file) or ".", exist_ok=True)
    with open(cookie_file, "w", encoding="utf-8") as f:
        json.dump(cookies, f, ensure_ascii=False, indent=4)
    print(f"Cookie已保存到: {cookie_file}")


async def load_cookies(context, cookie_file="xiaohongshu_cookies.json"):
    """从文件加载Cookie到浏览器"""
    if os.path.exists(cookie_file):
        with open(cookie_file, "r", encoding="utf-8") as f:
            cookies = json.load(f)
        await context.add_cookies(cookies)
        print(f"已从 {cookie_file} 加载Cookie")
        return True
    else:
        print(f"Cookie文件 {cookie_file} 不存在")
        return False


async def check_login_status(page):
    """检查是否已登录小红书"""
    # 检查是否存在头像元素或其他登录状态指示
    try:
        # 可能的登录状态指示元素: 用户头像或用户名
        avatar = page.locator("div.avatar").first
        if await avatar.count() > 0:
            print("已检测到登录状态")
            return True
    except Exception:
        pass

    # 检查是否存在"登录"按钮，如果存在则表示未登录
    try:
        login_button = page.locator("text=登录").first
        if await login_button.count() > 0:
            print("未检测到登录状态")
            return False
    except Exception:
        pass

    # 保守起见，如果无法确定，则假设未登录
    return False


async def manual_login(page, timeout=20):
    """等待用户手动登录并保存Cookie"""
    print(f"请在 {timeout} 秒内完成手动登录")
    await page.goto("https://www.xiaohongshu.com/")

    try:
        # 等待登录成功后出现的"我"导航元素
        await page.wait_for_selector('span.channel:text("我")', timeout=timeout * 1000)
        print("检测到成功登录")
        return True
    except Exception as e:
        print(f"登录等待超时或未检测到登录成功: {e}")
        return False


@server.tool()
async def extract_xiaohongshu_history(profile_id: str):
    """根据小红书用户profile_id, 提取小红书用户历史笔记

    Args:
        profile_id (str): 小红书用户profile_id

    Returns:
        list: 小红书用户历史笔记列表
    """

    force_login=False
    async with async_playwright() as p:
        # 启动浏览器(非headless模式)
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(viewport={"width": 1280, "height": 800})
        page = await context.new_page()

        # 尝试使用Cookie登录
        login_success = False
        if not force_login:
            if await load_cookies(context, cookie_file):
                # 访问首页检查登录状态
                await page.goto("https://www.xiaohongshu.com/")
                time.sleep(3)
                login_success = await check_login_status(page)

        # 如果Cookie登录失败，尝试手动登录
        if not login_success:
            print("Cookie登录失败，需要手动登录")
            login_success = await manual_login(page)
            if login_success:
                await save_cookies(context, cookie_file)
            else:
                print("手动登录失败，继续以未登录状态运行")

        # 访问用户主页
        user_url = f"https://www.xiaohongshu.com/user/profile/{profile_id}"
        await page.goto(user_url)
        print(f"正在访问用户主页: {user_url}")
        time.sleep(5)  # 等待页面加载

        # 获取用户名称
        username_element = page.locator("h1").first
        username = (
            username_element.text_content()
            if await username_element.count() > 0
            else "未知用户"
        )
        print(f"用户名称: {username}")

        # 确保点击"笔记"标签
        try:
            note_tab = page.locator("text=笔记").first
            await note_tab.click()
            print("已点击笔记标签")
            time.sleep(2)
        except Exception as e:
            print(f"点击笔记标签时出错: {e}")

        note_links = await page.locator("a.cover.mask.ld").all()
        if len(note_links) == 0:  # 备选方案1
            note_links = await page.locator("a[class*='cover mask']").all()
            if len(note_links) == 0:  # 备选方案2
                note_links = await page.locator(
                    "section.note-item a[href*='/user/profile/']"
                ).all()
                if len(note_links) == 0:  # 备选方案3
                    note_links = await page.locator(
                        "a[href*='/user/profile/']:not([class='name'])"
                    ).all()

        print(f"找到 {len(note_links)} 个可能的笔记链接")
        results = []
        max_notes = min(3, len(note_links))

        for i in range(max_notes):
            try:
                # 获取笔记链接
                note_link = await note_links[i].get_attribute("href")
                if not note_link.startswith("http"):
                    note_link = "https://www.xiaohongshu.com" + note_link

                print(f"正在提取第 {i+1}/{max_notes} 个笔记: {note_link}")

                # 打开新标签页访问笔记
                note_page = await context.new_page()
                await note_page.goto(note_link)
                time.sleep(3)  # 等待页面加载

                # 提取笔记标题
                title_element = note_page.locator("h1").first
                title = (
                    await title_element.text_content()
                    if await title_element.count() > 0
                    else "无标题"
                )

                # 提取笔记内容 - 基于截图中的class="note-text"
                content = ""
                try:
                    # 尝试多种可能的选择器
                    content_element = note_page.locator("span.note-text").first
                    if await content_element.count() > 0:
                        content = await content_element.text_content()
                    else:
                        # 备选选择器1
                        content_element = note_page.locator("div.desc span").first
                        if await content_element.count() > 0:
                            content = await content_element.text_content()
                        else:
                            # 备选选择器2，基于截图中的id="detail-desc"
                            content_element = note_page.locator("div#detail-desc").first
                            if await content_element.count() > 0:
                                content = await content_element.text_content()
                except Exception as e:
                    print(f"提取笔记内容时出错: {e}")
                print(f"提取到的内容: {content}")  # 只打印前50个字符
                # 保存笔记数据
                note_data = {
                    "title": json.dumps(title, ensure_ascii=False),
                    "content": json.dumps(content, ensure_ascii=False),
                }

                results.append(note_data)
                await note_page.close()

            except Exception as e:
                print(f"处理笔记时出错: {e}")

        await browser.close()
        return json.dumps(results, ensure_ascii=False)
    

if __name__ == "__main__":
    # res = asyncio.run(extract_xiaohongshu_history("5e2c390f0000000001007a10"))
    # res = asyncio.run(extract_xiaohongshu_history("5a13f37211be105003b83522"))
    # print(res)
    server.run()
