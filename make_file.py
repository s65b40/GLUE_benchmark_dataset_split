# -*- coding: utf-8 -*-
# author:Haochun Wang

import argparse
import os
import random
import shutil


# Copy the files for those that have original test set
def copy_with_test_set():
    # 2 SST-2   Train\Dev with header   Test without header
    # Note that there are two source folders
    shutil.copy('original/GLUE-SST-2/train.tsv', os.path.join('split/STS-2', 'train.tsv'))
    shutil.copyfile('original/GLUE-SST-2/dev.tsv', os.path.join('split/STS-2', 'dev.tsv'))
    shutil.copyfile('original/SST-2/test.tsv', os.path.join('split/STS-2', 'test.tsv'))
    print('2 SST-2 Copyfile done.')


    return


# Use Dev set as Test set and sample some from Training set as NEW Dev set.
def split_dev_from_training_set(args):

    try:
        os.mkdir('split')
    except:
        pass
    for folder in ['CoLA', 'STS-2', 'MRPC', 'STS-B', 'QQP', 'MNLI', 'QNLI', 'RTE', 'WNLI']:
        try:
            os.mkdir(os.path.join('split', folder))
        except:
            pass

    random.seed(args.seed)
    if args.copy:
        copy_with_test_set()
    else:
        print("No operation on those datasets that have original test sets.")

    # 1 CoLA            No header
    shutil.copyfile('original/CoLA/dev.tsv', os.path.join('split/CoLA/', 'test.tsv'))
    with open('original/CoLA/train.tsv', mode='r') as f_CoLA:
        lines_CoLA = f_CoLA.readlines()
        random.shuffle(lines_CoLA)
        with open('split/CoLA/dev.tsv', mode='w') as f_CoLA_new_dev:
            f_CoLA_new_dev.writelines(lines_CoLA[:args.sampled_from_train_CoLA])
        with open('split/CoLA/train.tsv', mode='w') as f_CoLA_new_train:
            f_CoLA_new_train.writelines(lines_CoLA[args.sampled_from_train_CoLA:])
    print('1 CoLA Split done. Training: %s, Dev: %s.' % (str(len(lines_CoLA[args.sampled_from_train_CoLA:])),
                                                         str(len(lines_CoLA[:args.sampled_from_train_CoLA]))))
    
    # 3 MRPC    With header
    shutil.copyfile('original/MRPC/dev.tsv', os.path.join('split/MRPC/', 'test.tsv'))
    with open('original/MRPC/train.tsv', mode='r') as f_MRPC:
        lines_MRPC = f_MRPC.readlines()
        header_MRPC = lines_MRPC[0]
        lines_MRPC = lines_MRPC[1:]
        random.shuffle(lines_MRPC)
        with open('split/MRPC/dev.tsv', mode='w') as f_MRPC_new_dev:
            f_MRPC_new_dev.writelines(header_MRPC)
            f_MRPC_new_dev.writelines(lines_MRPC[:args.sampled_from_train_MRPC])
        with open('split/MRPC/train.tsv', mode='w') as f_MRPC_new_train:
            f_MRPC_new_train.writelines(header_MRPC)
            f_MRPC_new_train.writelines(lines_MRPC[args.sampled_from_train_MRPC:])
    print('3 MRPC Split done. Training: %s, Dev: %s.' % (str(len(lines_MRPC[args.sampled_from_train_MRPC:])),
                                                         str(len(lines_MRPC[:args.sampled_from_train_MRPC]))))

    # 4 STS-B   With header
    shutil.copyfile('original/STS-B/dev.tsv', os.path.join('split/STS-B/', 'test.tsv'))
    with open('original/STS-B/train.tsv', mode='r') as f_STS_B:
        lines_STS_B = f_STS_B.readlines()
        header_STS_B = lines_STS_B[0]
        lines_STS_B = lines_STS_B[1:]
        random.shuffle(lines_STS_B)
        with open('split/STS-B/dev.tsv', mode='w') as f_STS_B_new_dev:
            f_STS_B_new_dev.writelines(header_STS_B)
            f_STS_B_new_dev.writelines(lines_STS_B[:args.sampled_from_train_STS_B])
        with open('split/STS-B/train.tsv', mode='w') as f_STS_B_new_train:
            f_STS_B_new_train.writelines(header_STS_B)
            f_STS_B_new_train.writelines(lines_STS_B[args.sampled_from_train_STS_B:])
    print('4 STS_B Split done. Training: %s, Dev: %s.' % (str(len(lines_STS_B[args.sampled_from_train_STS_B:])),
                                                          str(len(lines_STS_B[:args.sampled_from_train_STS_B]))))
    
    # 5 QQP         With header     Large file
    shutil.copyfile('original/QQP/dev.tsv', os.path.join('split/QQP/', 'test.tsv'))
    with open('original/QQP/train.tsv', mode='r') as f_QQP:
        lines_QQP = f_QQP.readlines()
        header_QQP = lines_QQP[0]
        lines_QQP = lines_QQP[1:]
        random.shuffle(lines_QQP)
        with open('split/QQP/dev.tsv', mode='w') as f_QQP_new_dev:
            f_QQP_new_dev.writelines(header_QQP)
            f_QQP_new_dev.writelines(lines_QQP[:args.sampled_from_train_QQP])
        with open('split/QQP/train.tsv', mode='w') as f_QQP_new_train:
            f_QQP_new_train.writelines(header_QQP)
            f_QQP_new_train.writelines(lines_QQP[args.sampled_from_train_QQP:])
    print('5 QQP Split done. Training: %s, Dev: %s.' % (str(len(lines_QQP[args.sampled_from_train_QQP:])),
                                                        str(len(lines_QQP[:args.sampled_from_train_QQP]))))

    # 6 MNLI        With header     Matched & Mismatched
    # separate test set
    shutil.copyfile('original/MNLI/dev_matched.tsv', os.path.join('split/MNLI/', 'test_matched.tsv'))
    shutil.copyfile('original/MNLI/dev_mismatched.tsv', os.path.join('split/MNLI/', 'test_mismatched.tsv'))
    # merged test set
    with open('original/MNLI/dev_matched.tsv', mode='r') as f_MNLI_dev_matched:
        lines_MNLI_dev_matched = f_MNLI_dev_matched.readlines()
    with open('original/MNLI/dev_mismatched.tsv', mode='r') as f_MNLI_dev_mismatched:
        lines_MNLI_dev_mismatched = f_MNLI_dev_mismatched.readlines()
    MNLI_new_test = lines_MNLI_dev_matched + lines_MNLI_dev_mismatched[1:]
    with open('split/MNLI/test.tsv', mode='w') as f_MNLI_new_test:
        f_MNLI_new_test.writelines(MNLI_new_test)

    with open('original/MNLI/train.tsv', mode='r') as f_MNLI:
        lines_MNLI = f_MNLI.readlines()
        header_MNLI = lines_MNLI[0]
        lines_MNLI = lines_MNLI[1:]
        random.shuffle(lines_MNLI)
        with open('split/MNLI/dev.tsv', mode='w') as f_MNLI_new_dev:
            f_MNLI_new_dev.writelines(header_MNLI)
            f_MNLI_new_dev.writelines(lines_MNLI[:args.sampled_from_train_MNLI])
        with open('split/MNLI/train.tsv', mode='w') as f_MNLI_new_train:
            f_MNLI_new_train.writelines(header_MNLI)
            f_MNLI_new_train.writelines(lines_MNLI[args.sampled_from_train_MNLI:])
    print('6 MNLI Split done. Training: %s, Dev: %s.' % (str(len(lines_MNLI[args.sampled_from_train_MNLI:])),
                                                         str(len(lines_MNLI[:args.sampled_from_train_MNLI]))))

    # 7 QNLI    With header
    shutil.copyfile('original/QNLI/dev.tsv', os.path.join('split/QNLI/', 'test.tsv'))
    with open('original/QNLI/train.tsv', mode='r') as f_QNLI:
        lines_QNLI = f_QNLI.readlines()
        header_QNLI = lines_QNLI[0]
        lines_QNLI = lines_QNLI[1:]
        random.shuffle(lines_QNLI)
        with open('split/QNLI/dev.tsv', mode='w') as f_QNLI_new_dev:
            f_QNLI_new_dev.writelines(header_QNLI)
            f_QNLI_new_dev.writelines(lines_QNLI[:args.sampled_from_train_QNLI])
        with open('split/QNLI/train.tsv', mode='w') as f_QNLI_new_train:
            f_QNLI_new_train.writelines(header_QNLI)
            f_QNLI_new_train.writelines(lines_QNLI[args.sampled_from_train_QNLI:])
    print('7 QNLI Split done. Training: %s, Dev: %s.' % (str(len(lines_QNLI[args.sampled_from_train_QNLI:])),
                                                         str(len(lines_QNLI[:args.sampled_from_train_QNLI]))))

    # 8 RTE    With header
    shutil.copyfile('original/RTE/dev.tsv', os.path.join('split/RTE/', 'test.tsv'))
    with open('original/RTE/train.tsv', mode='r') as f_RTE:
        lines_RTE = f_RTE.readlines()
        header_RTE = lines_RTE[0]
        lines_RTE = lines_RTE[1:]
        random.shuffle(lines_RTE)
        with open('split/RTE/dev.tsv', mode='w') as f_RTE_new_dev:
            f_RTE_new_dev.writelines(header_RTE)
            f_RTE_new_dev.writelines(lines_RTE[:args.sampled_from_train_RTE])
        with open('split/RTE/train.tsv', mode='w') as f_RTE_new_train:
            f_RTE_new_train.writelines(header_RTE)
            f_RTE_new_train.writelines(lines_RTE[args.sampled_from_train_RTE:])
    print('8 RTE Split done. Training: %s, Dev: %s.' % (str(len(lines_RTE[args.sampled_from_train_RTE:])),
                                                        str(len(lines_RTE[:args.sampled_from_train_RTE]))))

    # 9 WNLI    With header
    shutil.copyfile('original/WNLI/dev.tsv', os.path.join('split/WNLI/', 'test.tsv'))
    with open('original/WNLI/train.tsv', mode='r') as f_WNLI:
        lines_WNLI = f_WNLI.readlines()
        header_WNLI = lines_WNLI[0]
        lines_WNLI = lines_WNLI[1:]
        random.shuffle(lines_WNLI)
        with open('split/WNLI/dev.tsv', mode='w') as f_WNLI_new_dev:
            f_WNLI_new_dev.writelines(header_WNLI)
            f_WNLI_new_dev.writelines(lines_WNLI[:args.sampled_from_train_WNLI])
        with open('split/WNLI/train.tsv', mode='w') as f_WNLI_new_train:
            f_WNLI_new_train.writelines(header_WNLI)
            f_WNLI_new_train.writelines(lines_WNLI[args.sampled_from_train_WNLI:])
    print('9 WNLI Split done. Training: %s, Dev: %s.' % (str(len(lines_WNLI[args.sampled_from_train_WNLI:])),
                                                         str(len(lines_WNLI[:args.sampled_from_train_WNLI]))))
    return


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=19)
    parser.add_argument('--copy', type=bool, default=True)
    parser.add_argument('--sampled_from_train_CoLA', type=int, default=1043)
    parser.add_argument('--sampled_from_train_MRPC', type=int, default=408)
    parser.add_argument('--sampled_from_train_STS_B', type=int, default=1500)
    parser.add_argument('--sampled_from_train_QQP', type=int, default=40430)
    parser.add_argument('--sampled_from_train_MNLI', type=int, default=9832)
    parser.add_argument('--sampled_from_train_QNLI', type=int, default=5463)
    parser.add_argument('--sampled_from_train_RTE', type=int, default=277)
    parser.add_argument('--sampled_from_train_WNLI', type=int, default=71)
    args_ = parser.parse_args()
    return args_


if __name__ == '__main__':
    args = parse_args()
    split_dev_from_training_set(args)

