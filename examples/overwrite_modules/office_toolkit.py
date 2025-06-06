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
        files_content = []
        
        if not file_paths:
            return files_content
        
        word = win32com.client.Dispatch("Word.Application")
        word.Visible = False

        for file_path in file_paths:
            doc = word.Documents.Open(file_path)
            
            try:
                # 主文档内容
                main_text = doc.Content.Text
                with open("output.txt", "w", encoding="utf-8") as f:
                    f.write(main_text) 
                print(main_text)#print长度有限  不能打印出全部字符
                print("\n")
                # 页眉
                header_text = []
                for section in doc.Sections:
                    for header in section.Headers:
                        header_text.append(header.Range.Text)
                print(header_text)
                print("\n")
                # 页脚
                footer_text = []
                for section in doc.Sections:
                    for footer in section.Footers:
                        footer_text.append(footer.Range.Text)
                print(footer_text)
                print("\n")
                # 文本框内容
                shape_text = []
                for shape in doc.Shapes:
                    if shape.TextFrame.HasText:
                        shape_text.append(shape.TextFrame.TextRange.Text)
                print(shape_text)
                print("\n")
                # 合并所有内容
                content = "\n".join([
                    "=== 正文 ===",
                    main_text,
                    "\n=== 页眉 ===",
                    "\n".join(header_text),
                    "\n=== 页脚 ===",
                    "\n".join(footer_text),
                    "\n=== 文本框 ===",
                    "\n".join(shape_text)
                ])
                files_content.append({"file_path": file_path, "content": content})
            finally:
                doc.Close(SaveChanges=False)
                word.Quit()
        return files_content
    

    def get_ppt_text(self, file_paths: list[str]):
        files_content = []
        
        if not file_paths:
            return files_content
        
        files_content = []

        powerpoint = win32com.client.Dispatch("PowerPoint.Application")
        
        for file_path in file_paths:
            try:
                # 打开
                pres = powerpoint.Presentations.Open(file_path, ReadOnly=True, WithWindow=False)

                text = []
                
                # 每页
                for slide in pres.Slides:
                    try:
                        # 标题
                        try:
                            title = slide.Shapes.Title
                            if title and hasattr(title, "TextFrame"):
                                text.append(title.TextFrame.TextRange.Text)
                        except AttributeError:
                            pass
                        
                        # 形状
                        for shape in slide.Shapes:
                            try:
                                if not hasattr(shape, "HasTextFrame") or not shape.HasTextFrame:
                                    continue
                                    
                                text_frame = shape.TextFrame
                                if hasattr(text_frame, "TextRange"):
                                    text.append(text_frame.TextRange.Text)
                                    
                                if hasattr(shape, "HasTable") and shape.HasTable:
                                    try:
                                        for row in range(1, shape.Table.Rows.Count + 1):
                                            for col in range(1, shape.Table.Columns.Count + 1):
                                                cell = shape.Table.Cell(row, col)
                                                if hasattr(cell, "Shape"):
                                                    text.append(cell.Shape.TextFrame.TextRange.Text)
                                    except:
                                        continue
                                        
                            except Exception as shape_error:
                                print(f"形状处理错误: {shape_error}")
                                continue
                    
                    except Exception as slide_error:
                        print(f"幻灯片处理错误: {slide_error}")
                        continue
                        
                files_content.append({"file_path": file_path, "content": "\n".join(filter(None, text))})
            
            except Exception as e:
                print(f"PPT处理失败: {e}")
                
            finally:
                pres.Close()

        powerpoint.Quit()          

        return files_content

    def get_excel_text(self, file_paths: list[str]):
        files_content = []
        
        if not file_paths:
            return files_content
        
        excel = win32com.client.Dispatch("Excel.Application")
        excel.Visible = False
        
        for file_path in file_paths:
            wb = excel.Workbooks.Open(file_path)
            text = []
            
            try:
                for sheet in wb.Sheets:
                    text.append(f"\n=== 工作表: {sheet.Name} ===")
                    used_range = sheet.UsedRange
                    for row in used_range.Rows:
                        row_text = []
                        for cell in row.Columns:
                            if cell.Value is not None:
                                row_text.append(str(cell.Value).strip())
                        text.append("\t".join(row_text))
                
                content = "\n".join(text)
                files_content.append({"file_path": file_path, "content": content})
            finally:
                wb.Close(SaveChanges=False)
        
        excel.Quit()

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
    files_content = office_toolkit.get_all_office_instances_content(files_info)
    from IPython import embed; embed()