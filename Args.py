# -*-coding:utf-8-*-
import os

# word_vector_args
embedding_path = os.path.join("Tencent AILab ChineseEmbedding", "Tencent_AILab_ChineseEmbedding.txt")
is_binary = False

# raw_file_args
raw_file_path = r"E:\研究生学习\text similarity\数据语料\证据关系对应文本数据"
temp_file_path = r"E:\研究生学习\text similarity\数据语料\temp"

# data
data_excel_content = r"E:\研究生学习\text similarity\文书内容.xls"
data_excel_tag = r"E:\研究生学习\text similarity\证据关系对应2.xls"

# 举证最小长度默认为6,质证为3
min_evidence_len = 6
min_opinion_len = 3
