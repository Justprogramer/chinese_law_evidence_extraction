#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import codecs
import sys

import torch
from sklearn.metrics import *


class Inference(object):

    def __init__(self, **kwargs):
        """
        Args:
            model: SLModel
            data_iter: DataIter
            path_conllu: str, conllu格式的文件路径
            path_result: str, 预测结果存放路径
            label2id_dict: dict({str: int})
        """
        for k in kwargs:
            self.__setattr__(k, kwargs[k])

        # id2label dict
        self.id2label_dict = dict()
        for label in self.label2id_dict:
            self.id2label_dict[self.label2id_dict[label]] = label

    def _get_inputs(self, feed_dict, use_cuda=True):
        feed_tensor_dict = dict()
        for feature_name in self.model.feature_names:
            tensor = self.tensor_from_numpy(  # [bs, max_len]
                feed_dict[feature_name], use_cuda=use_cuda)
            feed_tensor_dict[feature_name] = tensor
        if self.model.use_char:  # max_len_char
            char_tensor = self.tensor_from_numpy(
                feed_dict['char'], use_cuda=self.model.use_cuda)
            max_len = feed_dict[self.model.feature_names[0]].shape[-1]
            char_tensor = char_tensor.view(-1, max_len, self.data_iter.char_max_len)
            feed_tensor_dict['char'] = char_tensor
        return feed_tensor_dict

    def infer(self):
        """预测
        Returns:
            labels: list of int
        """
        self.model.eval()
        labels_pred = []
        for feed_dict in self.data_iter:
            feed_tensor_dict = self._get_inputs(feed_dict, self.model.use_cuda)

            logits = self.model(**feed_tensor_dict)
            # mask
            mask = feed_tensor_dict[str(self.model.feature_names[0])] > 0
            actual_lens = torch.sum(feed_tensor_dict[self.model.feature_names[0]] > 0, dim=1).int()
            label_ids_batch = self.model.predict(logits, actual_lens, mask)
            labels_pred.extend(self.id2label(label_ids_batch))  # list(list(int))
        return labels_pred

    def infer2file(self):
        """预测，将结果写入文件
        """
        self.model.eval()
        predict_label_all = []
        gold_label_all = []
        file_result = codecs.open(self.path_result, 'w', encoding='utf-8')
        test_data_reader = self.read_test_data()
        for feed_dict in self.data_iter:
            feed_tensor_dict = self._get_inputs(feed_dict, self.model.use_cuda)

            logits = self.model(**feed_tensor_dict)
            # mask
            mask = feed_tensor_dict[str(self.model.feature_names[0])] > 0
            actual_lens = torch.sum(feed_tensor_dict[self.model.feature_names[0]] > 0, dim=1).int()
            label_ids_batch = self.model.predict(logits, actual_lens, mask)
            labels_batch = self.id2label(label_ids_batch)  # list(list(int))

            # write to file
            batch_size = len(labels_batch)
            for i in range(batch_size):
                feature_items = test_data_reader.__next__()
                sent_len = len(feature_items[0])  # 句子实际长度
                predict_labels = labels_batch[i]
                gold_labels = feature_items[1]
                gold_label_all += gold_labels
                if len(predict_labels) < sent_len:  # 补全为`O`
                    print("predict_labels(%s)--> %s" % (len(predict_labels), predict_labels))
                    print("gold_labels(%s)--> %s" % (len(gold_labels), gold_labels))
                    predict_labels = predict_labels + ['O'] * (sent_len - len(predict_labels))
                predict_label_all += predict_labels
                for j in range(sent_len):
                    file_result.write(
                        '{0} {1} {2}\n'.format(' '.join(feature_items[0][j]), gold_labels[j], predict_labels[j]))
                file_result.write('\n')

            print('sentence: {0} / {1}\r'.format(self.data_iter.iter_variable, self.data_iter.data_count))
        print('sentence: {0} / {1}\n'.format(self.data_iter.data_count, self.data_iter.data_count))
        gold_label_all = self.label2id(gold_label_all)
        predict_label_all = self.label2id(predict_label_all)
        target_names = sorted(self.label2id_dict)
        # for tag in sorted(self.label2id_dict):
        #     if "-E" in tag:
        #         target_names.append("举证方")
        #     elif "-C" in tag:
        #         target_names.append("证据说明")
        #     elif "-T" in tag:
        #         target_names.append("证据")
        #     elif "-O" in tag:
        #         target_names.append("质证意见")
        #     elif "-A" in tag:
        #         target_names.append("质证方")
        #     else:
        #         target_names.append("其他")
        file_result.write(classification_report(gold_label_all, predict_label_all, target_names=target_names))
        # file_result.write("\nprecision: %s, recall: %s, f1: %s" % (
        #     precision_score(gold_label_all, predict_label_all, average="macro"),
        #     recall_score(gold_label_all, predict_label_all, average="macro"),
        #     f1_score(gold_label_all, predict_label_all, average="macro")
        # ))
        file_result.close()

    def id2label(self, label_ids_array):
        """将label ids转为label
        Args:
            label_ids_array: list(np.array)

        Returns:
            labels: list(list(str))
        """
        labels = []
        for label_array in label_ids_array:
            temp = []
            for idx in label_array:
                temp.append(self.id2label_dict[idx])
            labels.append(temp)
        return labels

    def label2id(self, labels_array):
        """将labels转为label ids
        Args:
            labels_array: array

        Returns:
            label_ids: array
        """
        label_ids = []
        for label in labels_array:
            label_ids.append(self.label2id_dict[label])
        return label_ids

    @staticmethod
    def tensor_from_numpy(data, dtype='long', use_cuda=True):
        """将numpy转换为tensor
        Args:
            data: numpy
            dtype: long or float
            use_cuda: bool
        """
        assert dtype in ('long', 'float')
        if dtype == 'long':
            data = torch.from_numpy(data).long()
        else:
            data = torch.from_numpy(data).float()
        if use_cuda:
            data = data.cuda()
        return data

    def read_test_data(self):
        for test in self.data_raw:
            yield test
