# -*-coding:utf-8-*-
import argparse
import codecs
import json
import os
import time

import docx2txt
import pandas as pd
from win32com import client as wc

import Args
import util.common_util as my_util

# 证据名称列表
evidence_list = list()
# 笔录正文字典 文件名:内容
content_dict = dict()
# 笔录中举证质证文本 文件名：内容
evidence_paragraph_dict = dict()
# 笔录中存在的证据对应关系 文件名:[举证方 Evidence(E) ,证据名称 Trigger(T) ,证实内容 Content(C), 质证意见 Opinion(O),质证方 Anti-Evidence(A)]
tag_dic = dict()
# 标签训练数据
train = list()
# 完整文档存储路径
content_path = os.path.join('..', os.path.join('data', 'content.json'))
# 完整标签存储路径
tag_path = os.path.join('..', os.path.join('data', 'tag.json'))
# 标签中出现的所有证据名称统计路径
evidence_path = os.path.join('..', os.path.join('data', 'evidence.json'))
# 文档中只包含举证质证段落存储路径
evidence_paragraph_path = os.path.join('..', os.path.join('data', 'evidence_paragraph.json'))
# 处理成句子标签格式存储路径
train_path = os.path.join('..', os.path.join('data', 'train.json'))

# 句子标签全为"O"的数量
o_count_sentence = 0
# 每个标签的数量
o_tag_count = 0
e_tag_count = 0
t_tag_count = 0
c_tag_count = 0
a_tag_count = 0
# other标签的数量
other_tag_count = 0


def analyse_data():
    analyse_data_excel_content()
    # analyse_dir_document()
    analyse_data_excel_tags()
    extract_evidence_paragraph()
    create_train_data()


def save():
    # 保存处理的数据
    dump_data(content_dict, content_path)
    dump_data(tag_dic, tag_path)
    dump_data(evidence_list, evidence_path)
    dump_data(evidence_paragraph_dict, evidence_paragraph_path)
    dump_data(train, train_path)


# 从excel中加载数据
def analyse_data_excel_content(title=None, content=None):
    if title is None and content is None:
        rows = pd.read_excel(Args.data_excel_content, sheet_name=0, header=0)
        for title, content in rows.values:
            title = my_util.format_brackets(title.strip())
            # print(title)
            analyse_data_excel_content(title, content)
    else:
        old_paragraphs = [paragraph for paragraph in my_util.split_paragraph(content)
                          if paragraph is not None and len(paragraph.strip()) > 0]
        new_paragraphs = list()
        new_paragraph = ""
        # 合并发言人段落
        for paragraph in old_paragraphs:
            if my_util.check_paragraph(paragraph):
                if new_paragraph is not None and len(new_paragraph) > 0:
                    new_paragraphs.append(new_paragraph)
                new_paragraph = paragraph
            else:
                new_paragraph = new_paragraph + paragraph
        content_dict[title] = [
            [my_util.clean_text(sentence) for sentence in paragraph.split("。")
             if sentence is not None and len(sentence.strip()) > 0]
            for paragraph in new_paragraphs]


# 从doc和docx中加载文档，暂不使用
def analyse_dir_document():
    listdir = os.listdir(Args.raw_file_path)
    if not os.path.exists(Args.temp_file_path):
        os.makedirs(Args.temp_file_path)
    for file_name in listdir:
        title = my_util.format_brackets(file_name.split(".")[0])
        if title not in content_dict:
            path = os.path.join(Args.raw_file_path, file_name)
            if "docx" in file_name or "DOCX" in file_name:
                pass
            else:
                try:
                    word = wc.Dispatch('Word.Application')
                    doc = word.Documents.Open(path)
                    path = os.path.join(Args.temp_file_path, "temp.docx")
                    doc.SaveAs(path, "16")  # 转化后路径下的文件
                    doc.Close()
                    word.Quit()
                except:
                    print("《%s》 发生错误" % title)
                    continue
            content = docx2txt.process(path)
            print("补充文档《%s》 成功" % title)
            analyse_data_excel_content(title, content)


# 举证方 Evidence(E) 证据名称 Trigger(T) 证实内容 Content(C) 质证意见 Opinion(O) 质证方 Anti-Evidence(A)
def analyse_data_excel_tags():
    rows = pd.read_excel(Args.data_excel_tag, sheet_name=0, header=0)
    for title, E, T, C, O, A in rows.values:
        title = my_util.clean_text(title)
        E = my_util.clean_text(E)
        T = my_util.clean_text(T)
        C = my_util.clean_text(C)
        O = my_util.clean_text(O)
        A = my_util.clean_text(A)
        title = my_util.format_brackets(title)
        # print("tag_title:%s" % title)
        T = [sentence for sentence in T.split("。") if sentence is not None and len(sentence.strip()) > 0]
        C = [sentence for sentence in C.split("。") if sentence is not None and len(sentence.strip()) > 0]
        O = [sentence for sentence in O.split("。") if sentence is not None and len(sentence.strip()) > 0]
        if title not in tag_dic:
            tag_list = list()
            for t in T:
                tag_list.append([E, t, C, O, A])
            tag_dic[title] = tag_list
        else:
            for t in T:
                tag_dic[title].append([E, t, C, O, A])
                if t not in evidence_list:
                    evidence_list.append(t)


# 抽取主要举证质证段落
def extract_evidence_paragraph():
    for d in content_dict:
        start, end = my_util.check_evidence_paragraph(content_dict[d])
        # print(
        #     "提取证据段落完成《%s》(%s)，起始位置：%s,结束位置：%s\n%s\n%s" % (
        #         d, len(content_dict[d]), start, end, content_dict[d][start],
        #         content_dict[d][end - 1]))
        evidence_paragraph_dict[d] = content_dict[d][start:end]


# 先调用load_data
def create_train_data():
    global o_count_sentence
    for d in evidence_paragraph_dict:
        if d not in tag_dic:
            with codecs.open("log.txt", "a", "utf-8") as f:
                f.write("文档《%s》没有对应的数据标签" % d)
            continue
        evidence_content = evidence_paragraph_dict[d]
        for content in evidence_content:
            for sentence in content:
                tag = ["O" for i in range(len(sentence))]
                is_change = False
                for [E, t, C, O, A] in tag_dic[d]:
                    find_e = str(sentence).find(E + "：")
                    if find_e != -1:
                        is_change = True
                        tag[find_e] = "B-E"
                        for i in range(find_e + 1, find_e + len(E)):
                            tag[i] = "I-E"
                    find_t = str(sentence).find(t)
                    if find_t != -1:
                        is_change = True
                        tag[find_t] = "B-T"
                        for i in range(find_t + 1, find_t + len(t)):
                            tag[i] = "I-T"
                    for c in C:
                        find_c = str(sentence).find(c)
                        if find_c != -1:
                            is_change = True
                            tag[find_c] = "B-C"
                            for i in range(find_c + 1, find_c + len(c)):
                                tag[i] = "I-C"
                    for o in O:
                        find_o = str(sentence).find(o)
                        if find_o != -1:
                            is_change = True
                            tag[find_o] = "B-O"
                            for i in range(find_o + 1, find_o + len(o)):
                                tag[i] = "I-O"
                    find_a = str(sentence).find(A + "：")
                    if find_a != -1:
                        is_change = True
                        tag[find_a] = "B-A"
                        for i in range(find_a + 1, find_a + len(A)):
                            tag[i] = "I-A"
                if not is_change:
                    o_count_sentence += 1
                else:
                    # 不保存全部为other标签的句子
                    train.append(([word for word in sentence], tag))
                # 保存所有情况的标签
                # train.append(([word for word in sentence], tag))


# 标签统计
def statistic_data():
    global o_tag_count, e_tag_count, t_tag_count, c_tag_count, a_tag_count, other_tag_count
    for sentence, tag in train:
        for t in tag:
            if t == "B-E":
                e_tag_count += 1
            if t == "B-T":
                t_tag_count += 1
            if t == "B-O":
                o_tag_count += 1
            if t == "B-C":
                c_tag_count += 1
            if t == "B-A":
                a_tag_count += 1
            if t == "O":
                other_tag_count += 1


# 保存数据
def dump_data(data, path):
    with codecs.open(path, "w", "utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False))


# 写入日志
def dump_log():
    with codecs.open("log.txt", "a", "utf-8") as f:
        f.write("excel文本数据：%s条\n" % len(content_dict))
        f.write("证据关系文本数量：%s条\n" % len(tag_dic))
        f.write("处理获取语料数量：%s条\n" % len(train))
        f.write("处理获取语料数量标签全为'Other'：%s条\n" % o_count_sentence)
        f.write("统计语料包含'质证方'：%s个\n" % e_tag_count)
        f.write("统计语料包含'证据名称'：%s个\n" % t_tag_count)
        f.write("统计语料包含'证明内容'：%s个\n" % c_tag_count)
        f.write("统计语料包含'质证意见'：%s个\n" % o_tag_count)
        f.write("统计语料包含'质证方'：%s个\n" % a_tag_count)
        f.write("统计语料包含'Other'：%s个\n" % other_tag_count)
        f.write("\n")


# 加载数据
def load_data(data_path):
    content = ""
    with codecs.open(data_path, "r", "utf-8") as f:
        for line in f:
            content = content + line
    return json.loads(content)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("process_type", type=str, help="数据处理的格式，load加载，create构建，第一次使用create")
    args = arg_parser.parse_args()
    with codecs.open("log.txt", "a", "utf-8") as fp:
        fp.write("开始处理数据：[%s]" % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    start = time.time()
    if args.process_type is "create":
        analyse_data()
        statistic_data()
        save()
        dump_log()
    else:
        if my_util.is_file_exist(content_path) \
                and my_util.is_file_exist(tag_path) \
                and my_util.is_file_exist(evidence_paragraph_path) \
                and my_util.is_file_exist(train_path):
            evidence_paragraph_dict = load_data(evidence_paragraph_path)
            content_dict = load_data(content_path)
            tag_dic = load_data(tag_path)
            # train = load_data(train_path)
            create_train_data()
            statistic_data()
            dump_log()
        else:
            analyse_data()
            create_train_data()
            statistic_data()
            save()
            dump_log()
    with codecs.open("log.txt", "a", "utf-8") as fp:
        fp.write("处理数据结束：[%s], 费时：[%s]" % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), time.time() - start))
