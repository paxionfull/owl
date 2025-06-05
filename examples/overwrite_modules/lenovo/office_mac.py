import subprocess
import json
from typing import List, Dict, Optional

def run_applescript(script: str) -> str:
    """执行 AppleScript 并返回结果"""
    try:
        process = subprocess.Popen(['osascript', '-e', script],
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE)
        output, error = process.communicate()
        if error:
            print(f"执行 AppleScript 时出错: {error.decode('utf-8')}")
            return ""
        return output.decode('utf-8').strip()
    except Exception as e:
        print(f"执行 AppleScript 失败: {str(e)}")
        return ""

def get_word_documents() -> List[Dict[str, str]]:
    """获取当前打开的 Word 文档"""
    script = '''
    tell application "Microsoft Word"
        set docList to {}
        if running then
            try
                repeat with doc in documents
                    set docName to name of doc
                    set docPath to full name of doc
                    set end of docList to docName & "|" & docPath
                end repeat
            end try
        end if
        return docList
    end tell
    '''
    
    result = run_applescript(script)
    documents = []
    
    if result:
        for line in result.split(', '):
            if '|' in line:
                name, path = line.strip().split('|')
                documents.append({
                    'name': name.strip(),
                    'path': path.strip()
                })
    
    return documents

def get_excel_workbooks() -> List[Dict[str, str]]:
    """获取当前打开的 Excel 工作簿"""
    script = '''
    tell application "Microsoft Excel"
        set wbList to {}
        if running then
            try
                repeat with wb in workbooks
                    set wbName to name of wb
                    set wbPath to full name of wb
                    set end of wbList to wbName & "|" & wbPath
                end repeat
            end try
        end if
        return wbList
    end tell
    '''
    
    result = run_applescript(script)
    workbooks = []
    
    if result:
        for line in result.split(', '):
            if '|' in line:
                name, path = line.strip().split('|')
                workbooks.append({
                    'name': name.strip(),
                    'path': path.strip()
                })
    
    return workbooks

def get_powerpoint_presentations() -> List[Dict[str, str]]:
    """获取当前打开的 PowerPoint 演示文稿"""
    script = '''
    tell application "Microsoft PowerPoint"
        set pptList to {}
        if running then
            try
                repeat with pres in presentations
                    set pptName to name of pres
                    set pptPath to full name of pres
                    set end of pptList to pptName & "|" & pptPath
                end repeat
            end try
        end if
        return pptList
    end tell
    '''
    
    result = run_applescript(script)
    presentations = []
    
    if result:
        for line in result.split(', '):
            if '|' in line:
                name, path = line.strip().split('|')
                presentations.append({
                    'name': name.strip(),
                    'path': path.strip()
                })
    
    return presentations

def check_all_office_apps():
    """检查所有正在运行的 Office 应用及其打开的文件"""
    print("\n检查 Office 应用状态...")
    
    # 检查 Word
    word_docs = get_word_documents()
    if word_docs:
        print("\n当前打开的 Word 文档:")
        for doc in word_docs:
            print(f"- 文档: {doc['name']} (路径: {doc['path']})")
    else:
        print("\n未检测到打开的 Word 文档")
    
    # 检查 Excel
    excel_books = get_excel_workbooks()
    if excel_books:
        print("\n当前打开的 Excel 工作簿:")
        for wb in excel_books:
            print(f"- 工作簿: {wb['name']} (路径: {wb['path']})")
    else:
        print("\n未检测到打开的 Excel 工作簿")
    
    # 检查 PowerPoint
    ppt_files = get_powerpoint_presentations()
    if ppt_files:
        print("\n当前打开的 PowerPoint 演示文稿:")
        for ppt in ppt_files:
            print(f"- 演示文稿: {ppt['name']} (路径: {ppt['path']})")
    else:
        print("\n未检测到打开的 PowerPoint 演示文稿")

def check_word_instances():
    """检查当前打开的 Word 文档"""
    docs = get_word_documents()
    if docs:
        print("\n当前打开的 Word 文档:")
        for doc in docs:
            print(f"- 文档: {doc['name']} (路径: {doc['path']})")
    else:
        print("\n未检测到打开的 Word 文档")

def check_excel_instances():
    """检查当前打开的 Excel 工作簿"""
    books = get_excel_workbooks()
    if books:
        print("\n当前打开的 Excel 工作簿:")
        for book in books:
            print(f"- 工作簿: {book['name']} (路径: {book['path']})")
    else:
        print("\n未检测到打开的 Excel 工作簿")

def check_ppt_instances():
    """检查当前打开的 PowerPoint 演示文稿"""
    ppts = get_powerpoint_presentations()
    if ppts:
        print("\n当前打开的 PowerPoint 演示文稿:")
        for ppt in ppts:
            print(f"- 演示文稿: {ppt['name']} (路径: {ppt['path']})")
    else:
        print("\n未检测到打开的 PowerPoint 演示文稿")

if __name__ == "__main__":
    # 检查所有 Office 应用
    check_all_office_apps()
    
    # 或者单独检查各个应用
    # check_word_instances()
    # check_excel_instances()
    # check_ppt_instances() 