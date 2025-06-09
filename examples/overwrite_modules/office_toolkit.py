import sys
sys.path.append("D:\\work\\联想demo\\owl")

from camel.toolkits import FunctionTool
from camel.toolkits.base import BaseToolkit
from camel.utils import MCPServer
from typing import List, Any, Dict, Optional
import win32com.client
import json


@MCPServer()
class OfficeToolkit(BaseToolkit):
    r"""A toolkit for reading, creating and managing Office documents.
    """

    def __init__(self):
        self.app_names = ["Word", "Excel", "PowerPoint"]
        self.app_instances = OfficeToolkit.get_office_instances(self.app_names)

    @staticmethod
    def get_office_instances(app_names: List[str]) -> Dict[str, Any]:
        instances = dict()
        for app_name in app_names:
            try:
                app = win32com.client.GetActiveObject(f"{app_name}.Application")
                instances[app_name] = app
            except:
                pass
        return instances
    
    def get_word_instances(self):
        r"""获取用户正在使用的Word文件路径.
        """
        app_instance = self.app_instances.get("Word")
        word_instances = []
        if app_instance:
            print("\n检测到的 Word 实例:")
            for doc in app_instance.Documents:
                print(f"- 文档: {doc.Name} (路径: {doc.FullName})")
                word_instances.append(doc.FullName)
        else:
            print("\n未检测到 Word 实例！")
        return word_instances

    def get_ppt_instances(self):
        r"""获取用户正在使用的PPT文件路径.
        """
        app_instance = self.app_instances.get("PowerPoint")
        ppt_instances = []
        if app_instance:
            print("\n检测到的 PowerPoint 实例:")
            for pres in app_instance.Presentations:
                print(f"- 演示文稿: {pres.Name} (路径: {pres.FullName})")
                ppt_instances.append(pres.FullName)
        else:
            print("\n未检测到 PowerPoint 实例！")
        return ppt_instances

    def get_excel_instances(self):
        r"""获取用户正在使用的Excel文件路径.
        """
        app_instance = self.app_instances.get("Excel")
        excel_instances = []
        if app_instance:
            print("\n检测到的 Excel 实例:")
            for wb in app_instance.Workbooks:
                print(f"- 工作簿: {wb.Name} (路径: {wb.FullName})")
                excel_instances.append(wb.FullName)
        else:
            print("\n未检测到 Excel 实例！")
        return excel_instances
    
    #获取word文本成功
    def get_word_text(self, file_paths: list[str]):
        """从已经打开的Word文档中读取内容，不打开新文档，不关闭现有文档"""
        files_content = []
        
        if not file_paths:
            return files_content
        
        # 只使用已存在的Word实例
        word_app = self.app_instances.get("Word")
        if not word_app:
            print("未检测到正在运行的Word实例")
            return files_content

        try:
            for file_path in file_paths:
                try:
                    # 在已打开的文档中查找目标文档
                    target_doc = None
                    for doc in word_app.Documents:
                        if doc.FullName.lower() == file_path.lower():
                            target_doc = doc
                            break
                    
                    if not target_doc:
                        print(f"文档 {file_path} 未在当前Word实例中找到")
                        continue
                    
                    # 提取文档内容
                    main_text = target_doc.Content.Text
                    
                    # 页眉
                    header_text = []
                    try:
                        for section in target_doc.Sections:
                            for header in section.Headers:
                                if header.Range.Text.strip():
                                    header_text.append(header.Range.Text.strip())
                    except Exception as e:
                        print(f"读取页眉时出错: {e}")
                    
                    # 页脚
                    footer_text = []
                    try:
                        for section in target_doc.Sections:
                            for footer in section.Footers:
                                if footer.Range.Text.strip():
                                    footer_text.append(footer.Range.Text.strip())
                    except Exception as e:
                        print(f"读取页脚时出错: {e}")
                    
                    # 文本框内容
                    shape_text = []
                    try:
                        for shape in target_doc.Shapes:
                            if hasattr(shape, 'TextFrame') and shape.TextFrame.HasText:
                                shape_text.append(shape.TextFrame.TextRange.Text)
                    except Exception as e:
                        print(f"读取文本框时出错: {e}")
                    
                    # 合并所有内容
                    content = "\n".join([
                        "=== 正文 ===",
                        main_text,
                        "\n=== 页眉 ===",
                        "\n".join(header_text) if header_text else "无",
                        "\n=== 页脚 ===", 
                        "\n".join(footer_text) if footer_text else "无",
                        "\n=== 文本框 ===",
                        "\n".join(shape_text) if shape_text else "无"
                    ])
                    
                    files_content.append({"file_path": file_path, "content": content})
                    print(f"成功读取Word文档: {file_path}")
                    
                except Exception as e:
                    print(f"处理Word文档 {file_path} 时出错: {e}")
                    continue
                    
        except Exception as e:
            print(f"访问Word实例时出错: {e}")
        
        return files_content
    

    def get_ppt_text(self, file_paths: list[str]):
        """从已经打开的PowerPoint文档中读取内容，不打开新文档，不关闭现有文档"""
        files_content = []
        
        if not file_paths:
            return files_content
        
        # 只使用已存在的PowerPoint实例
        ppt_app = self.app_instances.get("PowerPoint")
        if not ppt_app:
            print("未检测到正在运行的PowerPoint实例")
            return files_content

        try:
            for file_path in file_paths:
                try:
                    # 在已打开的演示文稿中查找目标文档
                    target_pres = None
                    for pres in ppt_app.Presentations:
                        if pres.FullName.lower() == file_path.lower():
                            target_pres = pres
                            break
                    
                    if not target_pres:
                        print(f"演示文稿 {file_path} 未在当前PowerPoint实例中找到")
                        continue

                    text = []
                    
                    # 遍历每一页幻灯片
                    for slide in target_pres.Slides:
                        try:
                            # 尝试获取标题
                            try:
                                if hasattr(slide.Shapes, 'Title') and slide.Shapes.Title:
                                    title = slide.Shapes.Title
                                    if hasattr(title, "TextFrame") and title.TextFrame.HasText:
                                        text.append(f"【幻灯片标题】{title.TextFrame.TextRange.Text}")
                            except:
                                pass
                            
                            # 遍历所有形状
                            for shape in slide.Shapes:
                                try:
                                    if hasattr(shape, "HasTextFrame") and shape.HasTextFrame:
                                        text_frame = shape.TextFrame
                                        if hasattr(text_frame, "TextRange") and text_frame.HasText:
                                            text.append(text_frame.TextRange.Text)
                                    
                                    # 处理表格
                                    if hasattr(shape, "HasTable") and shape.HasTable:
                                        try:
                                            for row in range(1, shape.Table.Rows.Count + 1):
                                                for col in range(1, shape.Table.Columns.Count + 1):
                                                    cell = shape.Table.Cell(row, col)
                                                    if hasattr(cell, "Shape") and cell.Shape.TextFrame.HasText:
                                                        text.append(cell.Shape.TextFrame.TextRange.Text)
                                        except:
                                            continue
                                            
                                except Exception as shape_error:
                                    print(f"处理形状时出错: {shape_error}")
                                    continue
                        
                        except Exception as slide_error:
                            print(f"处理幻灯片时出错: {slide_error}")
                            continue
                            
                    content = "\n".join(filter(None, text))
                    files_content.append({"file_path": file_path, "content": content})
                    print(f"成功读取PowerPoint文档: {file_path}")
                
                except Exception as e:
                    print(f"处理PowerPoint文档 {file_path} 时出错: {e}")
                    continue
        
        except Exception as e:
            print(f"访问PowerPoint实例时出错: {e}")

        return files_content

    def get_excel_text(self, file_paths: list[str]):
        """从已经打开的Excel文档中读取内容，不打开新文档，不关闭现有文档"""
        files_content = []
        
        if not file_paths:
            return files_content
        
        # 只使用已存在的Excel实例
        excel_app = self.app_instances.get("Excel")
        if not excel_app:
            print("未检测到正在运行的Excel实例")
            return files_content
        
        try:
            for file_path in file_paths:
                try:
                    # 在已打开的工作簿中查找目标文档
                    target_wb = None
                    for wb in excel_app.Workbooks:
                        if wb.FullName.lower() == file_path.lower():
                            target_wb = wb
                            break
                    
                    if not target_wb:
                        print(f"工作簿 {file_path} 未在当前Excel实例中找到")
                        continue
                    
                    text = []
                    
                    # 遍历所有工作表
                    for sheet in target_wb.Sheets:
                        try:
                            text.append(f"\n=== 工作表: {sheet.Name} ===")
                            
                            # 获取已使用的单元格范围
                            used_range = sheet.UsedRange
                            if used_range:
                                for row in range(1, used_range.Rows.Count + 1):
                                    row_text = []
                                    for col in range(1, used_range.Columns.Count + 1):
                                        cell_value = used_range.Cells(row, col).Value
                                        if cell_value is not None:
                                            row_text.append(str(cell_value).strip())
                                        else:
                                            row_text.append("")
                                    if any(row_text):  # 只添加非空行
                                        text.append("\t".join(row_text))
                        
                        except Exception as sheet_error:
                            print(f"处理工作表 {sheet.Name} 时出错: {sheet_error}")
                            continue
                    
                    content = "\n".join(text)
                    files_content.append({"file_path": file_path, "content": content})
                    print(f"成功读取Excel文档: {file_path}")
                
                except Exception as e:
                    print(f"处理Excel文档 {file_path} 时出错: {e}")
                    continue
        
        except Exception as e:
            print(f"访问Excel实例时出错: {e}")
        
        return files_content


    def get_all_office_instances(self):
        r"""检查用户正在使用的电脑上，有哪些office文档正在运行.
        office文档类型包括：Word, Excel, PowerPoint.

        Returns:
            str: 用户正在使用的office软件路径
        """
        word_paths = self.get_word_instances()
        ppt_paths = self.get_ppt_instances()
        excel_paths = self.get_excel_instances()

#         output = """
# 用户正在使用的office软件路径如下：
# Word: 
# {}
# PowerPoint: 
# {}
# Excel: 
# {}
# """.format('\n'.join(word_paths), '\n'.join(ppt_paths), '\n'.join(excel_paths))
#         print(output)
        files_info = {
            "Word": word_paths,
            "PowerPoint": ppt_paths,
            "Excel": excel_paths
        }
        # return files_info
        files_info_str = json.dumps(files_info, ensure_ascii=False)
        return files_info_str
    
    def get_all_office_instances_content(self, file_paths: list[str]):
        r"""根据office文档路径，获取其内容.
        office文档类型包括：Word, Excel, PowerPoint.

        Args:
            file_paths (list[str]): office文档路径列表，类型为list, 尽量包含所有正在运行的office文档

        Returns:
            str: 文档的内容
        """
        words_files = []
        ppts_files = []
        excels_files = []
        for file_path in file_paths:
            file_type = file_path.split(".")[-1]
            if file_type in ["docx", "doc"]:
                words_files.append(file_path)
            elif file_type in ["pptx", "ppt"]:
                ppts_files.append(file_path)
            elif file_type in ["xlsx", "xls", "csv"]:
                excels_files.append(file_path)
        words_content = self.get_word_text(words_files)
        ppts_content = self.get_ppt_text(ppts_files)
        excels_content = self.get_excel_text(excels_files)
        content_info = {
            "Word": words_content,
            "PowerPoint": ppts_content,
            "Excel": excels_content
        }
        content_info_str = json.dumps(content_info, ensure_ascii=False)
        return content_info_str
        

    def get_tools(self) -> List[FunctionTool]:
        r"""Returns a list of FunctionTool objects representing the
        functions in the toolkit.

        Returns:
            List[FunctionTool]: A list of FunctionTool objects
                representing the functions in the toolkit.
        """
        return [FunctionTool(self.get_all_office_instances),
                FunctionTool(self.get_all_office_instances_content)]


if __name__ == "__main__":
    office_toolkit = OfficeToolkit()
    files_info = office_toolkit.get_all_office_instances()
    file_paths = files_info["Word"] + files_info["PowerPoint"] + files_info["Excel"]
    files_content = office_toolkit.get_all_office_instances_content(file_paths)
    from IPython import embed; embed()