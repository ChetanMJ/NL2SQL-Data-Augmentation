import random
import json
import numpy as np
import jsonlines

agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
cond_ops = ['=', '>', '<', 'OP']

def add_edge(node_id1,node_id2,g_adj):
    # undirected graph
    prev_node1_adj = g_adj[node_id1]
    prev_node2_adj = g_adj[node_id2]
    prev_node1_adj.append(node_id2)
    prev_node2_adj.append(node_id1)
    g_adj[node_id1] = prev_node1_adj
    g_adj[node_id2] = prev_node2_adj
    return g_adj

def add_edge_directed(node_id1,node_id2,g_adj):
    # directed graph id1->id2
    prev_node1_adj = g_adj[node_id1]
    prev_node1_adj.append(node_id2)
    g_adj[node_id1] = prev_node1_adj
    return g_adj

def load_files(sql_path,table_path,out_path,flag):
    data_test = []
    table_test = []
    cnt_err = 0
    punctuation = '!,;:?"%\''
    with open(sql_path, "r+", encoding = 'utf-8') as f:
        for item in jsonlines.Reader(f):
            data_test.append(item)
    with open(table_path, "r+", encoding = 'utf-8') as f:
        for item in jsonlines.Reader(f):
            table_test.append(item)

    f_test = open(out_path,"w", encoding = 'utf-8')

    table_test_dict = {}
    for table in table_test:
        if table['id'] not in table_test_dict:
            table_test_dict[table['id']] = table['header']

    for example in data_test:
        g_ids = {}
        g_adj = {}
        info = {}
        g_id_feature = {}
        text = example['question']
        table_id = example['table_id']
        sql = example['sql']
        agg_type = agg_ops[int(sql['agg'])]
        sel_col = int(sql['sel'])
        conds = sql['conds']
        table_cols = table_test_dict[table_id]
        sel_col_name = table_cols[sel_col]
        node_cou = 0
        g_ids[str(node_cou)] = node_cou
        g_id_feature[str(node_cou)] = 'select'
        g_adj[str(node_cou)] = []

        node_cou += 1
        g_ids[str(node_cou)] = node_cou
        g_id_feature[str(node_cou)] = sel_col_name
        g_adj[str(node_cou)] = []
        if flag == "undirected":
            g_adj = add_edge(str(node_cou),"0",g_adj)
        else:
            g_adj = add_edge_directed(str(node_cou),"0",g_adj)

        if agg_type != "":
            node_cou += 1
            g_ids[str(node_cou)] = node_cou
            g_id_feature[str(node_cou)] = agg_type
            g_adj[str(node_cou)] = []
            if flag == "undirected":
                g_adj = add_edge("1",str(node_cou),g_adj)
            else:
                g_adj = add_edge_directed("1",str(node_cou),g_adj)

        node_cou += 1
        g_ids[str(node_cou)] = node_cou
        g_id_feature[str(node_cou)] = 'and'
        g_adj[str(node_cou)] = []

        if flag == "undirected":
            g_adj = add_edge(str(node_cou),"0",g_adj)
        else:
            g_adj = add_edge_directed(str(node_cou),"0",g_adj)

        and_id = node_cou

        cond_val_cou = 0
        text_val = text.strip()
        for cond in conds:
            col_id = cond[0]
            cond_op = cond_ops[cond[1]]
            cond_val = cond[2]
            cond_col = table_cols[col_id]
            node_cou += 1
            g_ids[str(node_cou)] = node_cou
            g_id_feature[str(node_cou)] = cond_col
            g_adj[str(node_cou)] = []
            if flag == "undirected":
                g_adj = add_edge(str(node_cou),str(and_id),g_adj)
            else:
                g_adj = add_edge_directed(str(node_cou),str(and_id),g_adj)
            node_cou += 1
            g_ids[str(node_cou)] = node_cou
            g_id_feature[str(node_cou)] = cond_op+"val"+str(cond_val_cou)
            g_adj[str(node_cou)] = []
            if flag == "undirected":
                g_adj = add_edge(str(node_cou),str(node_cou-1),g_adj)
            else:
                g_adj = add_edge_directed(str(node_cou),str(node_cou-1),g_adj)

            # text_val_list = text_val.split(' ')
            # text_val_tmp = ""

            # for token in text_val_list:
            #     token_lower = token.lower()
            #     before_pun = ""
            #     after_pun = ""
            #     for idx in range(len(punctuation)):
            #         if len(token_lower) < 2:
            #             continue
            #         if token_lower[0] == punctuation[idx]:
            #             token_lower = token_lower[1:]
            #             before_pun = punctuation[idx]
            #         if token_lower[-1] == punctuation[idx]:
            #             token_lower = token_lower[:-1]
            #             after_pun = punctuation[idx]
            #     token_lower_list = token_lower.strip().split(' ')
            #     if token_lower.strip() == str(cond_val).lower().strip():
            #         text_val_tmp += before_pun+"val"+str(cond_val_cou)+after_pun+" "
            #     else:
            #         text_val_tmp += token + " "
            # text_val_tmp = text_val_tmp.strip()
            # if text_val == text_val_tmp:
            #     print(text_val)
            #     print(str(cond_val).lower().strip())
            #     print(text_val_list)
            # text_val = text_val_tmp.strip()
            # cond_val_cou += 1
            text_val_lower = text_val.lower()
            target_token = str(cond_val).lower().strip()
            start_idx = text_val_lower.find(" "+target_token+" ")
            if start_idx == -1:
                start_idx = text_val_lower.find("\""+target_token+"\"")
            if start_idx == -1:
                start_idx = text_val_lower.find("\""+target_token+",")
            if start_idx == -1:
                start_idx = text_val_lower.find('\''+target_token+'\'')
            if start_idx == -1:
                start_idx = text_val_lower.find(" "+target_token+"?")
            if start_idx == -1:
                start_idx = text_val_lower.find("\""+target_token+"?")
            if start_idx == -1:
                start_idx = text_val_lower.find(" "+target_token+"\'")
            if start_idx == -1:
                start_idx = text_val_lower.find(" "+target_token+",")
            if start_idx == -1:
                start_idx = text_val_lower.find(target_token+",")
            if start_idx == -1:
                start_idx = text_val_lower.find(" "+target_token+"\n")
            if start_idx == -1:
                start_idx = text_val_lower.find("#"+target_token)
            if start_idx == -1:
                start_idx = text_val_lower.find("("+target_token+")")
            if start_idx == -1:
                start_idx = text_val_lower.find("$"+target_token)
            if start_idx == -1:
                start_idx = text_val_lower.find(" "+target_token)
            if start_idx == -1:
                start_idx = text_val_lower.find(target_token+" ")
            if start_idx == -1:
                start_idx = text_val_lower.find(target_token+"?")
            end_idx = start_idx + len(str(cond_val).lower().strip())
            
            if start_idx == -1:
                g_id_feature[str(node_cou)] = str(cond_val)
                cnt_err += 1
            else:
                text_val = text_val[0:start_idx] + " val"+str(cond_val_cou) + text_val[end_idx+1:]
            cond_val_cou += 1
            

        info["seq"] = text_val
        info["text"] = text.strip()
        info["g_ids"] = g_ids
        info['g_ids_features'] = g_id_feature
        info['g_adj'] = g_adj
        info['table_id'] = table_id
        
        f_test.write(json.dumps(info)+"\n")
    print(cnt_err)
    print(len(data_test))

if __name__ == "__main__":

    load_files("./test.jsonl","./test.tables.jsonl","./test_un.data",'undirected')
    load_files("./train.jsonl","./train.tables.jsonl","./train_un.data",'undirected')
    load_files("./dev.jsonl","./dev.tables.jsonl","./dev_un.data",'undirected')

    load_files("./test.jsonl","./test.tables.jsonl","./test.data",'directed')
    load_files("./train.jsonl","./train.tables.jsonl","./train.data",'directed')
    load_files("./dev.jsonl","./dev.tables.jsonl","./dev.data",'directed')

    pass