import sys
sys.path.append("D:\\work\\联想demo\\owl")

from camel.toolkits import FunctionTool
from camel.toolkits.base import BaseToolkit
from camel.utils import MCPServer
from typing import List, Any, Dict, Optional
import win32com.client
import json
import os
import datetime


@MCPServer()
class EmailToolkit(BaseToolkit):
    r"""A toolkit for reading, creating and managing emails.
    """

    def __init__(self):
        self.app_names = ["Outlook"]
        self.app_instances = EmailToolkit.get_email_instances(self.app_names)
    
    @staticmethod
    def get_email_instances(app_names: List[str]) -> Dict[str, Any]:
        instances = dict()
        for app_name in app_names:
            try:
                app = win32com.client.GetActiveObject(f"{app_name}.Application")
                instances[app_name] = app
            except:
                pass
        return instances

    def get_outlook_instance(self):
        r"""获取用户正在使用的Outlook实例.
        """
        outlook_instance = self.app_instances.get("Outlook").GetNamespace("MAPI")
        if not outlook_instance:
            print("\n未检测到 Outlook 实例！")
        return outlook_instance
    
    def get_meetings_on_specific_day(self, start_date: str, end_date: str):
        """获取指定时间范围内的会议

        Args:
            start_date (str): 开始日期, 格式为"YYYY-MM-DD"
            end_date (str): 结束日期, 格式为"YYYY-MM-DD"

        Returns:
            str: 会议信息列表
        """
        outlook = self.get_outlook_instance()
        calendar = outlook.GetDefaultFolder(9)  # 9 表示日历文件夹

        if isinstance(start_date, str):
            start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d').replace(hour=0, minute=0, second=0)
        if isinstance(end_date, str):
            end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d').replace(hour=23, minute=59, second=59)

        # 构建筛选条件（Outlook 要求的日期格式）
        restriction = (
            "[Start] >= '" + start_date.strftime('%m/%d/%Y %H:%M %p') + "' AND "
            "[End] <= '" + end_date.strftime('%m/%d/%Y %H:%M %p') + "' "
        )

        # 获取符合条件的会议
        meetings = calendar.Items.Restrict(restriction)
        meetings.Sort("[Start]")  # 按开始时间排序

        # 提取会议信息
        meetings_list = []
        for meeting in meetings:
            meetings_list.append({
                "Subject": str(meeting.Subject),
                "Start": str(meeting.Start),
                "End": str(meeting.End),
                "Organizer": str(meeting.Organizer),
                "RequiredAttendees": str(meeting.RequiredAttendees),
                "Location": str(meeting.Location),
            })

        return json.dumps(meetings_list, ensure_ascii=False)

    def get_emails_by_recipient_and_date(self, recipient_email: str,start_date: str,end_date: str):
        """获取指定收件人，指定时间范围内的邮件内容

        Args:
            recipient_email (str): 收件人邮箱
            start_date (str): 开始日期, 格式为"YYYY-MM-DD"
            end_date (str): 结束日期, 格式为"YYYY-MM-DD"

        Returns:
            str: 邮件信息列表
        """
        outlook = self.get_outlook_instance()
        inbox = outlook.GetDefaultFolder(6)  # 收件箱

        if isinstance(start_date, str):
            start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d').replace(hour=0, minute=0, second=0)
        if isinstance(end_date, str):
            end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d').replace(hour=23, minute=59, second=59)

        restriction = (
            "[ReceivedTime] >= '" + start_date.strftime('%m/%d/%Y %H:%M %p') + "' AND "
            "[ReceivedTime] <= '" + end_date.strftime('%m/%d/%Y %H:%M %p') + "' "
        )
    

        result_m =[]
        filtered_emails = inbox.Items.Restrict(restriction)
        for mail in filtered_emails:

            if hasattr(mail, 'To'):
                rr = mail.SenderName   #发件人
                to = mail.To
                if recipient_email.lower() in to.lower():
                    received_time = mail.ReceivedTime
                    
                    result_m.append({
                    "Subject": str(getattr(mail, 'Subject', '无主题')),
                    "Time": str(received_time.strftime('%Y-%m-%d %H:%M')),
                    "Sender": str(getattr(mail, 'SenderName', '未知发件人')),
                    "CC": str(getattr(mail, 'CC', '')),
                    "To": str(getattr(mail, 'To', '')),    #收件人   收件人地址需要通过 MailItem.To/.CC/.BCC 访问。
                    "Body Preview": (str(getattr(mail, 'Body', ''))[:50] + "...") if getattr(mail, 'Body', None) else "无正文"
                })
            if hasattr(mail, 'CC'):
                rr = mail.SenderName   #发件人
                cc = mail.CC
                if recipient_email.lower() in cc.lower():
                    print(recipient_email)
                    received_time = mail.ReceivedTime                   
                    result_m.append({
                    "Subject": str(getattr(mail, 'Subject', '无主题')),
                    "Time": str(received_time.strftime('%Y-%m-%d %H:%M')),
                    "Sender": str(getattr(mail, 'SenderName', '未知发件人')),
                    "To": str(getattr(mail, 'To', '')), 
                    "CC": str(getattr(mail, 'CC', '')),    #收件人   收件人地址需要通过 MailItem.To/.CC/.BCC 访问。
                    "Body Preview": (str(getattr(mail, 'Body', ''))[:50] + "...") if getattr(mail, 'Body', None) else "无正文"
                })        
        return json.dumps(result_m, ensure_ascii=False)

    def get_emails_by_date(self, start_date: str, end_date: str):
        """获取指定时间范围内的邮件内容

        Args:
            start_date (str): 开始日期, 格式为"YYYY-MM-DD"
            end_date (str): 结束日期, 格式为"YYYY-MM-DD"

        Returns:
            str: 邮件信息列表
        """
        outlook = self.get_outlook_instance()
        
        inbox = outlook.GetDefaultFolder(6)  # 收件箱

        if isinstance(start_date, str):
            start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d').replace(hour=0, minute=0, second=0)
        if isinstance(end_date, str):
            end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d').replace(hour=23, minute=59, second=59)
    
        restriction = (
            "[ReceivedTime] >= '" + start_date.strftime('%m/%d/%Y %H:%M %p') + "' AND "
            "[ReceivedTime] <= '" + end_date.strftime('%m/%d/%Y %H:%M %p') + "' "
        )
        emails = inbox.Items.Restrict(restriction)
        
        #meetings = inbox.Items.Restrict(restriction)
        emails.Sort("[ReceivedTime]")  # 按开始时间排序
            # 提取会议信息
        meetings_list = []
        for email in emails:
            received_time = email.ReceivedTime
            if isinstance(received_time, datetime.datetime):
                received_time = received_time.replace(tzinfo=None)
            elif isinstance(received_time, (str, bytes)):
        
                received_time = datetime.datetime.strptime(received_time, '%m/%d/%Y %I:%M %p')

            meetings_list.append({
                    "Subject": str(getattr(email, 'Subject', '无主题')),
                    "Time": received_time.strftime('%Y-%m-%d %H:%M'),
                    "Sender": str(getattr(email, 'SenderName', '未知发件人')),
                    "To": str(getattr(email, 'To', '')),
                    "Days Ago": (datetime.datetime.now() - received_time).days,
                    # "Body Preview": (str(getattr(email, 'Body', ''))[:50] + "...") if getattr(email, 'Body', None) else "无正文"
                    "Body": (str(getattr(email, 'Body', ''))) if getattr(email, 'Body', None) else "无正文"
                })

        return json.dumps(meetings_list, ensure_ascii=False)
    
    def get_outlook_attachments(self, start_date: str, end_date: str):
        """下载指定时间范围内邮件的所有附件到本地

        Args:
            start_date (str): 开始日期, 格式为"YYYY-MM-DD"
            end_date (str): 结束日期, 格式为"YYYY-MM-DD"

        Returns:
            str: 附件信息列表
        """
        # 连接到Outlook
        outlook = self.get_outlook_instance()

        # 将日期字符串转换为datetime对象,并设置时间为当天开始和结束
        if isinstance(start_date, str):
            start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d').replace(hour=0, minute=0, second=0)
        if isinstance(end_date, str):
            end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d').replace(hour=23, minute=59, second=59)

        # 获取收件箱
        inbox = outlook.GetDefaultFolder(6)  # 6代表收件箱
        restriction = (
            "[ReceivedTime] >= '" + start_date.strftime('%m/%d/%Y %H:%M %p') + "' AND "
            "[ReceivedTime] <= '" + end_date.strftime('%m/%d/%Y %H:%M %p') + "' "
        )
    

        result_m =[]
        filtered_emails = inbox.Items.Restrict(restriction)
        # 创建保存附件的目录
        save_path = os.path.join(os.getenv('USERPROFILE'), 'Outlook_Attachments')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        attachments_info = []
        
        for message in filtered_emails:
            if message.Attachments.Count > 0:
                for attachment in message.Attachments:
                    try:
                        # 构建附件保存路径
                        file_path = os.path.join(save_path, attachment.FileName)
                        
                        # 保存附件
                        attachment.SaveAsFile(file_path)
                        
                        # 记录附件信息
                        attachments_info.append({
                            'mail_subject': str(message.Subject),
                            'mail_sender': str(message.SenderEmailAddress),
                            'mail_date': str(message.ReceivedTime),
                            'attachment_name': str(attachment.FileName),
                            'attachment_path': str(file_path),
                            'attachment_size': str(attachment.Size)
                        })
                        
                    except Exception as e:
                        print(f"保存附件失败: {attachment.FileName}, 错误: {str(e)}")
        
        return json.dumps(attachments_info, ensure_ascii=False)
    
    def get_tools(self) -> List[FunctionTool]:
        r"""Returns a list of FunctionTool objects representing the
        functions in the toolkit.

        Returns:
            List[FunctionTool]: A list of FunctionTool objects
                representing the functions in the toolkit.
        """
        return [FunctionTool(self.get_meetings_on_specific_day),
                FunctionTool(self.get_emails_by_date),
                FunctionTool(self.get_emails_by_recipient_and_date),
                # FunctionTool(self.get_outlook_attachments)
                ]

if __name__ == "__main__":
    email_toolkit = EmailToolkit()
    print(email_toolkit.get_tools())

    # print(email_toolkit.get_outlook_attachments("2025-06-01", "2025-06-10"))
    # print(email_toolkit.filter_recent_emails("2025-06-01", "2025-06-10"))
    print(email_toolkit.get_meetings_on_specific_day("2025-06-01", "2025-06-10"))

    from IPython import embed
    embed()
