# -*-coding:utf-8-*-
import sys
from optparse import OptionParser

import torch
import yaml
from torch import optim

from infer import Inference
from model.sequence_labeling_trainer import SLTrainer
from nn.dataset import *
from nn.modules import *
from util.common_util import *
from util.data_analyse_util import *
from util.embedding import *


def extract_feature_dict(feature_cols, feature_names, feature_dict, data_path,
                         sentence_lens=None, normalize=True, has_label=True):
    """从数据中统计特征
    Args:
        feature_cols: list(int), 特征的列数
        feature_names: list(str), 特征名称
        feature_dict: dict
        data_path: str ,路径
        sentence_lens: list, 用于记录句子长度
        normalize: bool, 是否标准化单词
        has_label: bool, 数据是否带有标签
    """
    data_idx = 0
    data = load_data(data_path)
    for i, train_list in enumerate(data):
        update_feature_dict(
            train_list, feature_dict, feature_cols, feature_names,
            normalize=normalize, has_label=has_label)
        sentence_lens.append(len(train_list[0]))
        data_idx += 1
    return data_idx, data


def update_feature_dict(train_list, feature_dict, feature_cols, feature_names,
                        normalize=True, has_label=True):
    """
    更新特征字典
    Args:
        train_list: list(list) [[word,pos,tag,fileName],...]
        feature_dict: dict
        feature_cols: list(int)
        feature_names: list(str)
        normalize: bool, 是否标准化单词
        has_label: bool
    """
    for i, col in enumerate(feature_cols):
        for token in train_list[col]:
            if normalize:
                token = normalize_word(token)
            feature_dict[feature_names[i]].update([token])
    if has_label:
        for label in train_list[2]:
            feature_dict['label'].add(label)


def pre_processing(configs):
    path_train = configs['data_params']['path_train']
    path_test = configs['data_params']['path_test'] if 'path_test' in configs['data_params'] else None

    feature_cols = configs['data_params']['feature_cols']
    feature_names = configs['data_params']['feature_names']
    min_counts = configs['data_params']['alphabet_params']['min_counts']
    root_alphabet = configs['data_params']['alphabet_params']['path']
    path_pretrain_list = configs['data_params']['path_pretrain']

    normalize = configs['word_norm']
    feature_dict = {}
    for feature_name in feature_names:
        feature_dict[feature_name] = Counter()
    feature_dict['label'] = set()
    sentence_lens = []
    # 处理训练、开发、测试数据
    print('读取文件...')
    data_count, train_data = extract_feature_dict(
        feature_cols, feature_names, feature_dict, path_train, sentence_lens,
        normalize=normalize, has_label=True, )
    print('`{0}`: {1}'.format(path_train, data_count))
    data_count, test_data = extract_feature_dict(
        feature_cols, feature_names, feature_dict, path_test, sentence_lens,
        normalize=normalize, has_label=True, )
    print('`{0}`: {1}'.format(path_test, data_count))

    # 构建label alphabet
    token2id_dict = dict()
    label2id_dict = dict()
    for label_idx, label in enumerate(sorted(feature_dict['label'])):
        label2id_dict[label] = label_idx + 1  # 从1开始编号
    token2id_dict['label'] = label2id_dict
    path_label2id_pkl = os.path.join(root_alphabet, 'label.pkl')
    if not is_file_exist(root_alphabet):
        os.makedirs(root_alphabet)
    dump_pkl_data(label2id_dict, path_label2id_pkl)

    # 构建特征alphabet
    for i, feature_name in enumerate(feature_names):
        feature2id_dict = dict()
        start_idx = 1
        for item in sorted(feature_dict[feature_name].items(), key=lambda d: d[1], reverse=True):
            if item[1] < min_counts[i]:
                continue
            feature2id_dict[item[0]] = start_idx
            start_idx += 1
        token2id_dict[feature_name] = feature2id_dict
        # write to file
        dump_pkl_data(feature2id_dict, os.path.join(root_alphabet, '{0}.pkl'.format(feature_name)))
    dump_pkl_data(token2id_dict, os.path.join(root_alphabet, 'token2id_dict.pkl'))

    # 构建embedding table
    print('抽取预训练词向量...')
    for i, feature_name in enumerate(feature_names):
        if path_pretrain_list[i]:
            print('特征`{0}`使用预训练词向量`{1}`:'.format(feature_name, path_pretrain_list[i]))
            word_embed_table, exact_match_count, fuzzy_match_count, unknown_count, total_count = build_word_embed(
                token2id_dict[feature_name], path_pretrain_list[i])
            print('\t精确匹配: {0} / {1}'.format(exact_match_count, total_count))
            print('\t模糊匹配: {0} / {1}'.format(fuzzy_match_count, total_count))
            print('\tOOV: {0} / {1}'.format(unknown_count, total_count))
            # write to file
            path_pkl = os.path.join(os.path.dirname(path_pretrain_list[i]), '{0}.embed.pkl'.format(feature_name))
            dump_pkl_data(word_embed_table, path_pkl)
    # 将token转成id
    # train
    train_data_tokens2id_dict = {}
    for sentence in train_data:
        for j, col in enumerate(feature_cols):
            feature_name = feature_names[j]
            array = tokens2id_array(sentence[col], token2id_dict[feature_name])
            if feature_name not in train_data_tokens2id_dict:
                train_data_tokens2id_dict[feature_name] = list()
            train_data_tokens2id_dict[feature_name].append(array)
        if "label" not in train_data_tokens2id_dict:
            train_data_tokens2id_dict["label"] = list()
        label_arr = tokens2id_array(sentence[2], token2id_dict['label'])
        train_data_tokens2id_dict['label'].append(label_arr)
    data_tokens2id_dict_path = os.path.join(os.path.dirname(path_train), 'train.token2id.pkl')
    dump_pkl_data(train_data_tokens2id_dict, data_tokens2id_dict_path)
    # test
    test_data_tokens2id_dict = {}
    for sentence in test_data:
        for j, col in enumerate(feature_cols):
            feature_name = feature_names[j]
            array = tokens2id_array(sentence[col], token2id_dict[feature_name])
            if feature_name not in test_data_tokens2id_dict:
                test_data_tokens2id_dict[feature_name] = list()
            test_data_tokens2id_dict[feature_name].append(array)
        if "label" not in test_data_tokens2id_dict:
            test_data_tokens2id_dict["label"] = list()
        label_arr = tokens2id_array(sentence[2], token2id_dict['label'])
        test_data_tokens2id_dict['label'].append(label_arr)
    data_tokens2id_dict_path = os.path.join(os.path.dirname(path_train), 'test.token2id.pkl')
    dump_pkl_data(test_data_tokens2id_dict, data_tokens2id_dict_path)


def init_model(configs):
    """初始化模型
    Returns:
        model: SLModel
    """
    use_char = configs['model_params']['use_char']

    feature_names = configs['data_params']['feature_names']
    # init feature alphabet size dict
    feature_size_dict = dict()
    root_alphabet = configs['data_params']['alphabet_params']['path']
    for feature_name in feature_names:
        alphabet = read_bin(os.path.join(root_alphabet, '{0}.pkl'.format(feature_name)))
        feature_size_dict[feature_name] = len(alphabet) + 1
    alphabet = read_bin(os.path.join(root_alphabet, 'label.pkl'))
    feature_size_dict['label'] = len(alphabet) + 1
    if use_char:
        alphabet = read_bin(os.path.join(root_alphabet, 'char.pkl'))
        feature_size_dict['char'] = len(alphabet) + 1

    # init feature dim size dict and pretrain embed dict
    path_pretrain_list = configs['data_params']['path_pretrain']
    embed_sizes = configs['model_params']['embed_sizes']
    feature_dim_dict = dict()
    for i, feature_name in enumerate(feature_names):
        feature_dim_dict[feature_name] = embed_sizes[i]
    pretrained_embed_dict = dict()
    for i, feature_name in enumerate(feature_names):
        if path_pretrain_list[i]:
            path_pkl = os.path.join(os.path.dirname(path_pretrain_list[i]), '{0}.embed.pkl'.format(feature_name))
            embed = read_bin(path_pkl)
            feature_dim_dict[feature_name] = embed.shape[-1]
            pretrained_embed_dict[feature_name] = embed
    if use_char:
        feature_dim_dict['char'] = configs['model_params']['char_dim']

    # init requires_grad_dict
    require_grads = configs['model_params']['require_grads']
    require_grad_dict = {}
    for i, feature_name in enumerate(feature_names):
        require_grad_dict[feature_name] = require_grads[i]
    if use_char:
        require_grad_dict['char'] = configs['model_params']['char_requires_grad']

    # init char parameters
    filter_sizes = configs['model_params']['conv_filter_sizes']
    filter_nums = configs['model_params']['conv_filter_nums']

    # init rnn parameters
    rnn_unit_type = configs['model_params']['rnn_type']
    num_rnn_units = configs['model_params']['rnn_units']
    num_layers = configs['model_params']['rnn_layers']
    bi_flag = configs['model_params']['bi_flag']

    use_crf = configs['model_params']['use_crf']

    # init other parameters
    dropout_rate = configs['model_params']['dropout_rate']
    average_batch = configs['model_params']['average_batch']
    deterministic = configs['model_params']['deterministic']
    use_cuda = configs['model_params']['use_cuda']

    # init model
    sl_model = SLModel(
        feature_names=feature_names, feature_size_dict=feature_size_dict, feature_dim_dict=feature_dim_dict,
        pretrained_embed_dict=pretrained_embed_dict, require_grad_dict=require_grad_dict, use_char=use_char,
        filter_sizes=filter_sizes, filter_nums=filter_nums, rnn_unit_type=rnn_unit_type, num_rnn_units=num_rnn_units,
        num_layers=num_layers, bi_flag=bi_flag, dropout_rate=dropout_rate, average_batch=average_batch,
        use_crf=use_crf, use_cuda=use_cuda)

    if deterministic:  # for deterministic
        torch.backends.cudnn.enabled = False

    use_cuda = configs['model_params']['use_cuda']
    if use_cuda:
        sl_model = sl_model.cuda()

    return sl_model


def init_train_data(configs):
    """初始化训练数据
    Returns:
        data_iter_train: DataIter
        data_iter_dev: DataIter
    """
    path_train = configs['data_params']['path_train']
    char_max_len = configs['model_params']['char_max_len']
    batch_size = configs['model_params']['batch_size']
    max_len_limit = configs['max_len_limit']

    features_names = configs['data_params']['feature_names']
    data_names = [name for name in features_names]
    use_char = configs['model_params']['use_char']
    if use_char:
        data_names.append('char')
    data_names.append('label')

    # load train data
    train_object_dict = read_bin(os.path.join(os.path.dirname(path_train), 'train.token2id.pkl'))
    train_count = len(train_object_dict[data_names[0]])

    # 拆分训练集
    data_utils = DataUtil(
        train_count, train_object_dict, data_names, use_char=use_char, char_max_len=char_max_len,
        batch_size=batch_size, max_len_limit=max_len_limit)
    data_iter_train, data_iter_dev = data_utils.split_dataset(proportions=(8, 2), shuffle=False)

    return data_iter_train, data_iter_dev


def init_test_data(configs):
    """初始化测试数据
    Returns:
        data_test: DataRaw
        data_iter_test: DataIter
    """
    path_train = configs['data_params']['path_train']
    char_max_len = configs['model_params']['char_max_len']
    batch_size = configs['model_params']['batch_size']
    max_len_limit = configs['max_len_limit']

    features_names = configs['data_params']['feature_names']
    data_names = [name for name in features_names]
    use_char = configs['model_params']['use_char']
    if use_char:
        data_names.append('char')
    data_names.append('label')

    # load test data
    test_object_dict = read_bin(os.path.join(os.path.dirname(path_train), 'test.token2id.pkl'))

    # 拆分训练集
    data_utils = DataUtil(
        len(test_object_dict["word"]), test_object_dict, data_names, use_char=use_char, char_max_len=char_max_len,
        batch_size=batch_size, max_len_limit=max_len_limit)
    [data_iter_test] = data_utils.split_dataset(proportions=(1,), shuffle=False)

    return data_iter_test


def init_optimizer(configs, model):
    """初始化optimizer
    Returns:
        optimizer
    """
    optimizer_type = configs['model_params']['optimizer']
    learning_rate = configs['model_params']['learning_rate']
    l2_rate = configs['model_params']['l2_rate']
    momentum = configs['model_params']['momentum']
    lr_decay = 0
    # 过滤不需要更新参数的
    parameters = filter(lambda p: p.requires_grad, model.parameters())

    if optimizer_type.lower() == "sgd":
        lr_decay = configs['model_params']['lr_decay']
        optimizer = optim.SGD(parameters, lr=learning_rate, momentum=momentum, weight_decay=l2_rate)
    elif optimizer_type.lower() == "adagrad":
        optimizer = optim.Adagrad(parameters, lr=learning_rate, weight_decay=l2_rate)
    elif optimizer_type.lower() == "adadelta":
        optimizer = optim.Adadelta(parameters, lr=learning_rate, weight_decay=l2_rate)
    elif optimizer_type.lower() == "rmsprop":
        optimizer = optim.RMSprop(parameters, lr=learning_rate, weight_decay=l2_rate)
    elif optimizer_type.lower() == "adam":
        optimizer = optim.Adam(parameters, lr=learning_rate, weight_decay=l2_rate)
    else:
        print('请选择正确的optimizer: {0}'.format(optimizer_type))
        exit()
    return optimizer, lr_decay


def init_trainer(configs, data_iter_train, data_iter_dev, model, optimizer, lr_decay):
    """初始化model trainer
    Returns:
        trainer: SLTrainer
    """
    feature_names = configs['data_params']['feature_names']
    use_char = configs['model_params']['use_char']
    max_len_char = configs['model_params']['char_max_len']
    path_save_model = configs['data_params']['path_model']
    if not is_file_exist(path_save_model):
        os.makedirs(path_save_model)

    nb_epoch = configs['model_params']['nb_epoch']
    max_patience = configs['model_params']['max_patience']

    learning_rate = configs['model_params']['learning_rate']

    trainer = SLTrainer(
        data_iter_train=data_iter_train, data_iter_dev=data_iter_dev, feature_names=feature_names,
        use_char=use_char, max_len_char=max_len_char, model=model, optimizer=optimizer,
        path_save_model=path_save_model, nb_epoch=nb_epoch, max_patience=max_patience,
        learning_rate=learning_rate, lr_decay=lr_decay)

    return trainer


def load_model(configs):
    """加载预训练的model
    """
    model = init_model(configs)

    path_model = os.path.join(configs['data_params']['path_model'], "sequence_model")
    model_state = torch.load(path_model)
    model.load_state_dict(model_state)
    return model


def train_model(configs):
    """训练模型
    """
    # init model
    sl_model = init_model(configs)
    print(sl_model)

    # init data
    data_iter_train, data_iter_dev = init_train_data(configs)

    # init optimizer
    optimizer, lr_decay = init_optimizer(configs, model=sl_model)

    # init trainer
    model_trainer = init_trainer(
        configs, data_iter_train, data_iter_dev, sl_model, optimizer, lr_decay)

    model_trainer.fit()


def test_model(configs):
    """测试模型
    """
    path_test = configs['data_params']['path_test'] if 'path_test' in configs['data_params'] else None
    # init model
    model = load_model(configs)

    # init test data
    data_test = load_data(path_test)
    data_iter_test = init_test_data(configs)

    # init infer
    if 'path_test_result' not in configs['data_params'] or \
            not configs['data_params']['path_test_result']:
        path_result = configs['data_params']['path_test'] + '.result'
    else:
        path_result = configs['data_params']['path_test_result']
    # label to id dict
    path_pkl = os.path.join(configs['data_params']['alphabet_params']['path'], 'label.pkl')
    label2id_dict = read_bin(path_pkl)
    infer = Inference(
        model=model, data_iter=data_iter_test, data_raw=data_test,
        path_result=path_result, label2id_dict=label2id_dict)

    # do infer
    infer.infer2file()


def parse_opts():
    op = OptionParser()
    op.add_option(
        '-c', '--config', dest='config', type='str', help='配置文件路径')
    op.add_option('--train', dest='train', action='store_true', default=True, help='训练模式')
    op.add_option('--test', dest='test', action='store_true', default=False, help='测试模式')
    op.add_option(
        '-p', '--preprocess', dest='preprocess', action='store_true', default=False, help='是否进行预处理')
    argv = [] if not hasattr(sys.modules['__main__'], '__file__') else sys.argv[1:]
    (opts, args) = op.parse_args(argv)
    if not opts.config:
        op.print_help()
        exit()
    if opts.test:
        opts.train = False
    return opts


def main():
    opts = parse_opts()
    configs = yaml.load(codecs.open(opts.config, encoding='utf-8'), Loader=None)

    if opts.train:  # train
        # 判断是否需要预处理
        if opts.preprocess:
            pre_processing(configs)
            print("pre process 结束，开始训练模型。。。。")
        # 训练
        train_model(configs)
        print("训练模型结束，开始测试。。。。")
    # test
    test_model(configs)


if __name__ == '__main__':
    main()
