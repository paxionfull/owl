import win32com.client
import pythoncom
def get_office_instances(app_name):
    instances = []
    try:
        app = win32com.client.GetActiveObject(f"{app_name}.Application")
        instances.append(app)
    except:
        pass
    
    try:
        app = win32com.client.Dispatch(f"{app_name}.Application")
        if app not in instances:
            instances.append(app)
    except:
        pass
    
    return instances

def check_word_instances():
    word_instances = get_office_instances("Word")
    if word_instances:
        print("\n检测到的 Word 实例:")
        for word in word_instances:
            for doc in word.Documents:
                print(f"- 文档: {doc.Name} (路径: {doc.FullName})")
    else:
        print("\n未检测到 Word 实例！")

def check_ppt_instances():
    ppt_instances = get_office_instances("PowerPoint")
    if ppt_instances:
        print("\n检测到的 PowerPoint 实例:")
        for ppt in ppt_instances:
            for pres in ppt.Presentations:
                print(f"- 演示文稿: {pres.Name} (路径: {pres.FullName})")
    else:
        print("\n未检测到 PowerPoint 实例！")

def check_excel_instances():
    excel_instances = get_office_instances("Excel")
    if excel_instances:
        print("\n检测到的 Excel 实例:")
        for excel in excel_instances:
            for wb in excel.Workbooks:
                print(f"- 工作簿: {wb.Name} (路径: {wb.FullName})")
    else:
        print("\n未检测到 Excel 实例！")
def check_all_office_apps():
   
    #  Word
    word_instances = get_office_instances("Word")
    if word_instances:
        print("\n检测到的 Word 实例:")
        for word in word_instances:
            for doc in word.Documents:
                print(f"- 文档: {doc.Name} (路径: {doc.FullName})")
    else:
        print("\n未检测到 Word 实例！")

    #  ppt
    ppt_instances = get_office_instances("PowerPoint")
    if ppt_instances:
        print("\n检测到的 PowerPoint 实例:")
        for ppt in ppt_instances:
            for pres in ppt.Presentations:
                print(f"- 演示文稿: {pres.Name} (路径: {pres.FullName})")
    else:
        print("\n未检测到 PowerPoint 实例！")

    #  Excel
    excel_instances = get_office_instances("Excel")
    if excel_instances:
        print("\n检测到的 Excel 实例:")
        for excel in excel_instances:
            for wb in excel.Workbooks:
                print(f"- 工作簿: {wb.Name} (路径: {wb.FullName})")
    else:
        print("\n未检测到 Excel 实例！")
        
#获取word文本成功
def get_full_word_text(file_path):
    
    word = win32com.client.Dispatch("Word.Application")
    word.Visible = False
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
        return "\n".join([
            "=== 正文 ===",
            main_text,
            "\n=== 页眉 ===",
            "\n".join(header_text),
            "\n=== 页脚 ===",
            "\n".join(footer_text),
            "\n=== 文本框 ===",
            "\n".join(shape_text)
        ])
    finally:
        doc.Close(SaveChanges=False)
        word.Quit()


def get_full_word_text2():
    result = []
    word_instances = get_office_instances("Word")
    if word_instances:
        print("\n检测到的 Word 实例:")
        for word in word_instances:
            for doc in word.Documents:
                print(f"- 文档: {doc.Name} (路径: {doc.FullName})")
                filename = doc.Name
                filepath = doc.FullName
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
                content= "\n".join([
                    "=== 正文 ===",
                    main_text,
                    "\n=== 页眉 ===",
                    "\n".join(header_text),
                    "\n=== 页脚 ===",
                    "\n".join(footer_text),
                    "\n=== 文本框 ===",
                    "\n".join(shape_text)
                ])
                #print(content)
                result.append({
                    'filename': filename,
                    'filepath': filepath,
                    'content': content
                })
        return result
    else:
        print("\n未检测到 Word 实例！")
        return " "



def get_ppt_text(ppt_path):
    powerpoint = None
    pres = None
    text = []
    
    try:
        
        pythoncom.CoInitialize()
        
        powerpoint = win32com.client.Dispatch("PowerPoint.Application")
        
        # 打开
        pres = powerpoint.Presentations.Open(ppt_path, ReadOnly=True, WithWindow=False)
        
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
                
        return "\n".join(filter(None, text))
    
    except Exception as e:
        print(f"PPT处理失败: {e}")
        return None
        
    finally:
        try:
            if pres:
                pres.Close()
        except:
            pass
            
        try:
            if powerpoint:
                powerpoint.Quit()
        except:
            pass
            
  
        pythoncom.CoUninitialize()


def get_ppt_text2():
    pres = None
    
    result =[]
    pythoncom.CoInitialize()
    try:
        ppt_instances = get_office_instances("PowerPoint")
        if ppt_instances:
            print("\n检测到的 PowerPoint 实例:")
            for ppt in ppt_instances:
                for pres in ppt.Presentations:
                    print(f"- 演示文稿: {pres.Name} (路径: {pres.FullName})")
                    filename = pres.Name
                    filepath = pres.FullName
                    text = []
                    for slide in pres.Slides:
                        try:
                            # 标题
                            try:
                                if hasattr(slide.Shapes, 'HasTitle') and slide.Shapes.HasTitle:
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
                    result.append({
                        'filename': filename,
                        'filepath': filepath,
                        'content': text
                    })
                
    finally:
        pythoncom.CoUninitialize()
        return result


def get_excel_text(file_path):
    
    excel = win32com.client.Dispatch("Excel.Application")
    excel.Visible = False
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
        
        return "\n".join(text)
    finally:
        wb.Close(SaveChanges=False)
        excel.Quit()


def get_excel_text2():


    excel_instances = get_office_instances("Excel")
    result = []
    if excel_instances:
        print("\n检测到的 Excel 实例:")
        for excel in excel_instances:
            for wb in excel.Workbooks:
                print(f"- 工作簿: {wb.Name} (路径: {wb.FullName})")
                filename = wb.Name
                filepath = wb.FullName
                text = []
                for sheet in wb.Sheets:
                    text.append(f"\n=== 工作表: {sheet.Name} ===")
                    used_range = sheet.UsedRange
                    for row in used_range.Rows:
                        row_text = []
                        for cell in row.Columns:
                            if cell.Value is not None:
                                row_text.append(str(cell.Value).strip())
                        text.append("\t".join(row_text))
                result.append({
                    'filename': filename,
                    'filepath': filepath,
                    'content': text
                })
                #return "\n".join(text)
            return result
    else:
        print("\n未检测到 Excel 实例！")



# 使用示例  成功可用
#text = get_excel_text_win32com("D:\\服务器状态.xlsx")
#

#
if __name__ == "__main__":
    # get_excel_text2()
    # get_ppt_text2()
    # get_full_word_text2()

    # #获取正在打开的word，ppt，excel的名字和路径
    # check_all_office_apps()
    # #获取正在打开的word的名字和路径
    # check_word_instances()
    # #获取正在打开的ppt的名字和路径
    # check_ppt_instances()
    # #获取正在打开的excel的名字和路径
    # check_excel_instances()

    #获取word文字
    #text = get_full_word_text("D:\\test.docx")
    #print(text)

    text = get_ppt_text(r"C:\Users\Device Intelligent\workspace\assets\Document_API  sample\Document sample\Generate_template-Manus AI.pptx")
    #print(text)

    # text = get_excel_text("D:\\服务器状态.xlsx")
    # text = get_excel_text(r"C:\Users\Device Intelligent\workspace\assets\Document_API  sample\Document sample\刘丹-学期发展指导报告.xlsx")
    from IPython import embed
    embed()