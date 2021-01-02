import random
import json
import numpy as np
import jsonlines


def load_files(data_path,question_path,out_path):
    data_test = []
    question_test = []
    err_cnt = 0
    with open(question_path, "r+", encoding = 'utf-8') as f:
        for item in f.readlines():
            question_test.append(item)
    with open(data_path, "r+", encoding = 'utf-8') as f:
        lines = f.readlines()
        for line in lines:
            item = json.loads(line)
            data_test.append(item)
    print(len(data_test))
    print(len(question_test))
    f_test = open(out_path,"w", encoding = 'utf-8')
    for i in range(len(question_test)):
        flag = True
        info = {}
        question_i = question_test[i].strip()
        question_i_list = question_i.split(' ')
        for token in question_i_list:
            if token == '<oov>':
                flag = False
        if flag == False:
            err_cnt += 1
            continue
        info['phase'] = 2
        info['table_id'] = data_test[i]['table_id']
        info['question'] = question_i
        sql_new = {}
        sql_new['conds'] = data_test[i]['sql']['conds']
        sql_new['sel'] = data_test[i]['sql']['sel']
        sql_new['agg'] = data_test[i]['sql']['agg']
        info['sql'] = sql_new
        f_test.write(json.dumps(info)+"\n")
    print(err_cnt)
if __name__ == "__main__":

    load_files("./augmented_train.jsonl","./augment_data_sentences.txt","./augmented_train_final.jsonl")
    pass