# -*-coding:utf-8-*-
import platform
import re
import os


# 将中文括号替换成英文括号
def format_brackets(str):
    if str is None or is_nan(str):
        return ""
    replace = str.replace('（', '(').replace('）', ')')
    return clean_text(replace)


# 判断文本是否是发言起始段落
def check_paragraph(str):
    pattern_str = r"([原|被|代|审|书|时|地|案|公][\S]{0,4}|^[\S]{0,1})[：|:]"
    return re.search(pattern_str, str)


# 检测是否是windows平台
def is_windows():
    return platform.system() == "Windows"


def split_paragraph(str):
    return re.split(r'[\n|\r]{0,2}', str)


def clean_text(str):
    if str is None or is_nan(str):
        return ""
    return re.sub("[\r\n\\s]+", "", str)


def is_file_exist(path):
    return os.path.exists(path)


def is_nan(num):
    return num != num


def check_evidence_paragraph(document):
    evidence_start_pattern = r"审.*(?:被告|原告){0,1}.*(?:提供|举示|出示){0,1}.*(?:质证|证据|举证)"
    anti_evidence_start_patter = r"审.*[《|》]?.*(?:规定|责任|义务)"
    evidence_end_pattern = r"(?:质证|证据|举证|调查).*(?:结束|完毕)"
    start = 0
    for index, paragraph in enumerate(document):
        paragraph = "。".join(paragraph)
        if start == 0 \
                and re.search(evidence_start_pattern, paragraph) \
                and not re.search(anti_evidence_start_patter, paragraph):
            start = index
        if re.search(evidence_end_pattern, paragraph):
            end = index
            if end <= start:
                print("提取证据段落出错，起始位置：%s,结束位置：%s" % (start, end))
            else:
                return start, end
    return start, len(document) - 1


if __name__ == '__main__':
    str = "ab\nc\r  \td    "
    print(clean_text(str))
