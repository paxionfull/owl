
import win32com.client
import datetime
import os
def get_meetings_on_specific_day(start_date,end_date):
    outlook = win32com.client.Dispatch("Outlook.Application").GetNamespace("MAPI")
    calendar = outlook.GetDefaultFolder(9)  # 9 表示日历文件夹


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
            "Subject": meeting.Subject,
            "Start": meeting.Start,
            "End": meeting.End,
            "Organizer": meeting.Organizer,
            "RequiredAttendees": meeting.RequiredAttendees,
            "Location": meeting.Location,
        })

    return meetings_list

def filter_emails_by_recipient_and_date(recipient_email,start_date,end_date):
    outlook = win32com.client.Dispatch("Outlook.Application")
    namespace = outlook.GetNamespace("MAPI")
    inbox = namespace.GetDefaultFolder(6)  # 收件箱

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
                "Subject": getattr(mail, 'Subject', '无主题'),
                "Time": received_time.strftime('%Y-%m-%d %H:%M'),
                "Sender": getattr(mail, 'SenderName', '未知发件人'),
                "CC": getattr(mail, 'CC', ''),
                "To": getattr(mail, 'To', ''),    #收件人   收件人地址需要通过 MailItem.To/.CC/.BCC 访问。
                "Body Preview": (getattr(mail, 'Body', '')[:50] + "...") if getattr(mail, 'Body', None) else "无正文"
            })
        if hasattr(mail, 'CC'):
            rr = mail.SenderName   #发件人
            cc = mail.CC
            if recipient_email.lower() in cc.lower():
                print(recipient_email)
                received_time = mail.ReceivedTime                   
                result_m.append({
                "Subject": getattr(mail, 'Subject', '无主题'),
                "Time": received_time.strftime('%Y-%m-%d %H:%M'),
                "Sender": getattr(mail, 'SenderName', '未知发件人'),
                "To": getattr(mail, 'To', ''), 
                "CC": getattr(mail, 'CC', ''),    #收件人   收件人地址需要通过 MailItem.To/.CC/.BCC 访问。
                "Body Preview": (getattr(mail, 'Body', '')[:50] + "...") if getattr(mail, 'Body', None) else "无正文"
            })        
    return result_m

def filter_recent_emails(start_date,end_date):
    outlook = win32com.client.Dispatch("Outlook.Application").GetNamespace("MAPI")
    
    inbox = outlook.GetDefaultFolder(6)  # 收件箱
   
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
                "Subject": getattr(email, 'Subject', '无主题'),
                "Time": received_time.strftime('%Y-%m-%d %H:%M'),
                "Sender": getattr(email, 'SenderName', '未知发件人'),
                "To": getattr(email, 'To', ''),
                "Days Ago": (datetime.datetime.now() - received_time).days,
                "Body Preview": (getattr(email, 'Body', '')[:50] + "...") if getattr(email, 'Body', None) else "无正文"
            })

    return meetings_list
def get_outlook_attachments(start_date,end_date):
    # 连接到Outlook
    outlook = win32com.client.Dispatch("Outlook.Application").GetNamespace("MAPI")
    
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
                        'mail_subject': message.Subject,
                        'mail_sender': message.SenderEmailAddress,
                        'mail_date': message.ReceivedTime,
                        'attachment_name': attachment.FileName,
                        'attachment_path': file_path,
                        'attachment_size': attachment.Size
                    })
                    
                except Exception as e:
                    print(f"保存附件失败: {attachment.FileName}, 错误: {str(e)}")
    
    return attachments_info
if __name__ == "__main__":
    #附件
    start_date = datetime.datetime(2025, 5, 1, 0, 0, 0)  # 2025-05-14 00:00:00
    end_date = datetime.datetime(2025, 10, 1, 23, 59, 59)  # 2025-05-14 23:59:59
    attachments = get_outlook_attachments(start_date,end_date)
    for att in attachments:
        print(f"邮件主题: {att['mail_subject']}")
        print(f"发件人: {att['mail_sender']}")
        print(f"日期: {att['mail_date']}")
        print(f"附件名: {att['attachment_name']}")
        print(f"保存路径: {att['attachment_path']}")
        print(f"大小: {att['attachment_size']} 字节\n")
#用日期过滤会议
    meetings = get_meetings_on_specific_day(start_date,end_date)
    for idx, mtg in enumerate(meetings, 1):
        print(f"会议 {idx}: {mtg['Subject']}")
        print(f"时间: {mtg['Start']} - {mtg['End']}")
        print(f"组织者: {mtg['Organizer']}")
        print(f"参会人: {mtg['RequiredAttendees']}")
        print(f"地点: {mtg['Location']}\n")


    #用日期，收件人过滤邮件
    start_date = datetime.datetime(2025, 5, 1, 0, 0, 0)  # 2025-05-14 00:00:00
    end_date = datetime.datetime(2025, 10, 1, 23, 59, 59)  # 2025-05-14 23:59:59
    emails = filter_emails_by_recipient_and_date("li li13 zhou",start_date,end_date)

    for email in emails:
        print(email["Subject"])
        print("\n")
        print(email["Time"])
        print("\n")
        print(email["To"])
        print("\n")
        print(email["CC"])
        print("\n")