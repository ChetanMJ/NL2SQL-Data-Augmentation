<<<<<<< HEAD
import random
import json
import numpy as np
import jsonlines

sql_keywords = ["symselect","symagg","symcol","symwhere","symop","symcond","symand","symend","-lrb-","-rrb-",'--']
sql_words = ["select","agg","column","where","operator","condition","and","end","","",""]

def load_files(data_path,sql_path,out_path,flag):
    data_test = []
    sql_test = []
    with open(sql_path, "r+", encoding = 'utf-8') as f:
        for item in jsonlines.Reader(f):
            sql_test.append(item)
    with open(data_path, "r+", encoding = 'utf-8') as f:
        lines = f.readlines()
        for line in lines:
            item = json.loads(line)
            data_test.append(item)
    f_test = open(out_path,"w", encoding = 'utf-8')
    sql_dict = {}
    error_cnt = 0
    for sql_item in sql_test:
        table_id = sql_item["table_id"]
        question_words = sql_item["question"]["words"]
        sql_text = sql_item["seq_output"]["words"]
        if table_id not in sql_dict:
            sql_list = []
            sql_list.append((table_id,question_words,sql_text,1))
            sql_dict[table_id] = sql_list
        else:
            sql_list = sql_dict[table_id]
            cnt = len(sql_list)
            sql_list.append((table_id,question_words,sql_text,cnt+1))
            sql_dict[table_id] = sql_list
    cnt_dict = {}
    for example in data_test:
        info = {}
        info["seq"] = example["seq"]
        info["text"] = example["text"]
        info["g_ids"] = example["g_ids"]
        info['g_ids_features'] = example['g_ids_features']
        info['g_adj'] = example['g_adj']
        sql_text = ""
        table_id = example["table_id"]
        if table_id not in sql_dict:
            error_cnt += 1
            continue
        sql_list = sql_dict[table_id]
        if table_id not in cnt_dict:
            cnt_dict[table_id] = 1
        else:
            cnt_table = cnt_dict[table_id]
            cnt_dict[table_id] = cnt_table+1
        for sql in sql_list:
            q_text = sql[2]
            q_cnt = sql[3]
            if cnt_dict[table_id] == q_cnt:
                sql_text = q_text
                break
        sql_text_final = []
        if flag == 'sym':
            for token in sql_text:
                match = False
                for keywords in sql_keywords:
                    if token == keywords:
                        match = True
                if match == False:
                    sql_text_final.append(token)
        else:
            for token in sql_text:
                match = -1
                for j in range(len(sql_keywords)):
                    if token == sql_keywords[j]:
                        match = j
                if match == -1:
                    sql_text_final.append(token)
                else:
                    sql_text_final.append(sql_words[match])
        info["sql"] = sql_text_final
        f_test.write(json.dumps(info)+"\n")
    print(error_cnt)
if __name__ == "__main__":

    load_files("./test.data","./test_sql.jsonl","./test_sym_sql.data",'sym')
    load_files("./train.data","./train_sql.jsonl","./train_sym_sql.data",'sym')
    load_files("./dev.data","./dev_sql.jsonl","./dev_sym_sql.data",'sym')

    load_files("./test.data","./test_sql.jsonl","./test_word_sql.data",'word')
    load_files("./train.data","./train_sql.jsonl","./train_word_sql.data",'word')
    load_files("./dev.data","./dev_sql.jsonl","./dev_word_sql.data",'word')

=======
import random
import json
import numpy as np
import jsonlines

sql_keywords = ["symselect","symagg","symcol","symwhere","symop","symcond","symand","symend","-lrb-","-rrb-",'--']
sql_words = ["select","agg","column","where","operator","condition","and","end","","",""]

def load_files(data_path,sql_path,out_path,flag):
    data_test = []
    sql_test = []
    with open(sql_path, "r+", encoding = 'utf-8') as f:
        for item in jsonlines.Reader(f):
            sql_test.append(item)
    with open(data_path, "r+", encoding = 'utf-8') as f:
        lines = f.readlines()
        for line in lines:
            item = json.loads(line)
            data_test.append(item)
    f_test = open(out_path,"w", encoding = 'utf-8')
    sql_dict = {}
    error_cnt = 0
    for sql_item in sql_test:
        table_id = sql_item["table_id"]
        question_words = sql_item["question"]["words"]
        sql_text = sql_item["seq_output"]["words"]
        if table_id not in sql_dict:
            sql_list = []
            sql_list.append((table_id,question_words,sql_text,1))
            sql_dict[table_id] = sql_list
        else:
            sql_list = sql_dict[table_id]
            cnt = len(sql_list)
            sql_list.append((table_id,question_words,sql_text,cnt+1))
            sql_dict[table_id] = sql_list
    cnt_dict = {}
    for example in data_test:
        info = {}
        info["seq"] = example["seq"]
        info["text"] = example["text"]
        info["g_ids"] = example["g_ids"]
        info['g_ids_features'] = example['g_ids_features']
        info['g_adj'] = example['g_adj']
        sql_text = ""
        table_id = example["table_id"]
        if table_id not in sql_dict:
            error_cnt += 1
            continue
        sql_list = sql_dict[table_id]
        if table_id not in cnt_dict:
            cnt_dict[table_id] = 1
        else:
            cnt_table = cnt_dict[table_id]
            cnt_dict[table_id] = cnt_table+1
        for sql in sql_list:
            q_text = sql[2]
            q_cnt = sql[3]
            if cnt_dict[table_id] == q_cnt:
                sql_text = q_text
                break
        sql_text_final = []
        if flag == 'sym':
            for token in sql_text:
                match = False
                for keywords in sql_keywords:
                    if token == keywords:
                        match = True
                if match == False:
                    sql_text_final.append(token)
        else:
            for token in sql_text:
                match = -1
                for j in range(len(sql_keywords)):
                    if token == sql_keywords[j]:
                        match = j
                if match == -1:
                    sql_text_final.append(token)
                else:
                    sql_text_final.append(sql_words[match])
        info["sql"] = sql_text_final
        f_test.write(json.dumps(info)+"\n")
    print(error_cnt)
if __name__ == "__main__":

    load_files("./test.data","./test_sql.jsonl","./test_sym_sql.data",'sym')
    load_files("./train.data","./train_sql.jsonl","./train_sym_sql.data",'sym')
    load_files("./dev.data","./dev_sql.jsonl","./dev_sym_sql.data",'sym')

    load_files("./test.data","./test_sql.jsonl","./test_word_sql.data",'word')
    load_files("./train.data","./train_sql.jsonl","./train_word_sql.data",'word')
    load_files("./dev.data","./dev_sql.jsonl","./dev_word_sql.data",'word')

>>>>>>> 0b4a20d265ef4c226b583b8290c5ce3eb04a7a9f
    pass