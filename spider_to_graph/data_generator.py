import json
import spacy
CLAUSE_KEYWORDS = ('select', 'from', 'where', 'group', 'order', 'limit', 'intersect', 'union', 'except')
JOIN_KEYWORDS = ('join', 'on', 'as')

WHERE_OPS = ('not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists')
UNIT_OPS = ('none', '-', '+', "*", '/')
AGG_OPS = ('none', 'max', 'min', 'count', 'sum', 'avg')
TABLE_TYPE = {
    'sql': "sql",
    'table_unit': "table_unit",
}
COND_OPS = ('and', 'or')
SQL_OPS = ('intersect', 'union', 'except')
ORDER_OPS = ('desc', 'asc')

nlp = spacy.blank("en")

def word_tokenize(sent):
    doc = nlp(sent)
    return [token.text.lower() for token in doc]

from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()


class SpiderGraph:
    def __init__(self, example, table_dict):
        self.g_ids = {}
        self.g_id_feature = {}
        self.g_adj = {}
        self.table_dict = table_dict
        self.db_id = example['db_id']
        self.text = example['question']
        self.text_toks = self.process_question_toks(list(map(lambda x : x.lower(), example['question_toks'])))
        self.query = example['query']
        self.query_toks = self.process_query_toks(example['query_toks'])
        self.sql = example['sql']
        self.add_sql(self.sql)

    def process_query_toks(self, query_toks):
        split_toks = []
        for t in query_toks:
            split_toks += t.split('.')
        nl_toks = []
        to_parsed_headers = self.table_dict[self.db_id]['to_parsed_headers']
        for t in split_toks:
            t = t.lower()
            if t in to_parsed_headers:
                nl_toks += to_parsed_headers[t].split(' ')
            else:
                nl_toks += t.split(' ')
        final_toks = [wordnet_lemmatizer.lemmatize(t.lower()) for t in nl_toks]
        return final_toks

    def process_question_toks(self, question_toks):
        return [wordnet_lemmatizer.lemmatize(t) for t in question_toks]

    def add_node(self, text):
        node_cnt = len(self.g_ids)
        self.g_ids[str(node_cnt)] = node_cnt
        self.g_id_feature[str(node_cnt)] = text
        self.g_adj[str(node_cnt)] = []
        return str(node_cnt)

    def add_edge(self, node_id1,node_id2):
        # undirected graph
        g_adj = self.g_adj
        prev_node1_adj = g_adj[node_id1]
        prev_node2_adj = g_adj[node_id2]
        prev_node1_adj.append(node_id2)
        prev_node2_adj.append(node_id1)
        g_adj[node_id1] = prev_node1_adj
        g_adj[node_id2] = prev_node2_adj
    #     return g_adj

    def add_edge_directed(self, node_id1, node_id2):
        # directed graph id1->id2
        g_adj = self.g_adj
        prev_node1_adj = g_adj[node_id1]
        prev_node1_adj.append(node_id2)
        g_adj[node_id1] = prev_node1_adj
    #     return g_adj
    
    def get_col_name(self, col_id):
        return self.table_dict[self.db_id]["column_names"][col_id][1]
        
    def add_col_unit(self, col_unit):
        agg_id = col_unit[0]
        col_id = col_unit[1]
        is_distinct = col_unit[2]
        root_nid = None
        col_nid = None
        if agg_id > 0: # not none
            agg_nid = self.add_node(AGG_OPS[agg_id])
            root_nid = agg_nid
            col_nid = self.add_node(self.get_col_name(col_id))
            self.add_edge_directed(root_nid, col_nid)
        else:
            col_nid = self.add_node(self.get_col_name(col_id))
            root_nid = col_nid
        if is_distinct:
            distinct_nid = self.add_node('distinct')
            self.add_edge_directed(col_nid, distinct_nid)
            
        return root_nid
        
    def add_val_unit(self, val_unit):
        unit_op = val_unit[0]
        col_unit1 = val_unit[1]
        col_unit2 = val_unit[2]
        root_nid = None
        if col_unit1 is None:
            print("Error: No col_unit1")
        if unit_op > 0: # col1, col2 case
            unit_op_nid = self.add_node(UNIT_OPS[unit_op])
            root_nid = unit_op_nid
            col_unit1_nid = self.add_col_unit(col_unit1)
            self.add_edge_directed(unit_op_nid, col_unit1_nid)
            if col_unit2 is not None: # col_unit2 is usually none
                col_unit2_nid = self.add_col_unit(col_unit2)
                self.add_edge_directed(unit_op_nid, col_unit2_nid)
        else: # col1 case
            root_nid = self.add_col_unit(col_unit1)
            
        return root_nid
            
    def add_select_clause(self, select_clause):
        # 'select': (isDistinct(bool), [(agg_id, val_unit), (agg_id, val_unit), ...])
        select_nid = self.add_node('select')
        is_distinct = select_clause[0]
        if is_distinct:
            distinct_nid = self.add_node('distinct')
            self.add_edge_directed(select_nid, distinct_nid)

        def add_agg_val(agg_val):
            agg_id = agg_val[0]
            val_unit = agg_val[1]
            root_nid = None
            if agg_id > 0: # not none
                agg_nid = self.add_node(AGG_OPS[agg_id])
                root_nid = agg_nid
                val_unit_nid = self.add_val_unit(val_unit)
                self.add_edge_directed(agg_nid, val_unit_nid)
            else:
                root_nid = self.add_val_unit(val_unit)
                
            return root_nid

        agg_vals = select_clause[1]
        for agg_val in agg_vals:
            agg_val_nid = add_agg_val(agg_val)
            self.add_edge_directed(select_nid, agg_val_nid)

        return select_nid
    
    def get_table_name(self, table_id):
        return self.table_dict[self.db_id]["table_names"][table_id]
    
    def add_table_unit(self, table_unit):
        # a table unit can be a col or a nested sql...
        table_type = table_unit[0]
        if table_type == 'sql':
            sql = table_unit[1]
            nested_sql_nid = self.add_sql(sql)
            return nested_sql_nid
        else:
            assert(table_type == "table_unit")
            table_id = table_unit[1]
            table_nid = self.add_node(self.get_table_name(table_id))
            return table_nid
    
    def add_cond_unit(self, cond_unit):
        not_op = cond_unit[0]
        op_id = cond_unit[1]
        val_unit = cond_unit[2]
        val1 = cond_unit[3]
        val2 = cond_unit[4]
        root_nid = None
        if not_op > 0:
            not_nid = self.add_node("not")
            root_nid = not_nid
        else:
            where_op = WHERE_OPS[op_id]
            op_nid = self.add_node(where_op)
            root_nid = op_nid
            val_unit_nid = self.add_val_unit(val_unit)
            self.add_edge_directed(op_nid, val_unit_nid)
            # val1: number(float)/string(str)/sql(dict)/list(col_unit)
            if isinstance(val1, float) or isinstance(val1, int) or isinstance(val1, str):
                val1_nid = self.add_node(str(val1).strip().lower())
            if isinstance(val1, list): # column
                val1_nid = self.add_col_unit(val1)
            if isinstance(val1, dict): # nested sql
                val1_nid = self.add_sql(val1)
            self.add_edge_directed(op_nid, val1_nid)
            # val2: for between only
            if val2 is not None:
                val2_nid = self.add_node(str(val2))
                self.add_edge_directed(op_nid, val2_nid)
                
        return root_nid
    
    def add_conditions(self, conditions):
        if len(conditions) == 1:
            return self.add_cond_unit(conditions[0])
        else:
            assert(len(conditions) > 1 and len(conditions) % 2 == 1)
            cond_op = conditions[1]
            cond_op_nid = self.add_node(cond_op)
            for j in range(0, len(conditions), 2):
                cond_unit_nid = self.add_cond_unit(conditions[j])
                self.add_edge_directed(cond_op_nid, cond_unit_nid)
            return cond_op_nid
    
    def add_from_clause(self, from_clause):
        table_units = from_clause['table_units']
        conds = from_clause['conds']
        from_nid = self.add_node('from')
        for table_unit in table_units:
            table_nid = self.add_table_unit(table_unit)
            self.add_edge_directed(from_nid, table_nid)

        if len(conds) > 0:
            on_nid = self.add_node('on')
            self.add_edge_directed(from_nid, on_nid)
            conds_nid = self.add_conditions(conds)
            self.add_edge_directed(on_nid, conds_nid)
        
        return from_nid
    
    def add_where_clause(self, conds):
        # assume where is not empty (checked outside function)
        where_nid = self.add_node('where')
        conds_nid = self.add_conditions(conds)
        self.add_edge_directed(where_nid, conds_nid)
        return where_nid
    
    def add_group_by_clause(self, col_units):
        group_by_nid = self.add_node('group by')
        for col_unit in col_units:
            col_unit_nid = self.add_col_unit(col_unit)
            self.add_edge_directed(group_by_nid, col_unit_nid)
        return group_by_nid
    
    def add_order_by_clause(self, order_by_clause):
        assert(len(order_by_clause) == 2)
        order_by_nid = self.add_node('order by')
        order = 'ascending' if order_by_clause[0] == 'asc' else 'descending' # desc/asc, use English words
        val_units = order_by_clause[1]
        order_nid = self.add_node(order)
        self.add_edge_directed(order_by_nid, order_nid)
        for val_unit in val_units:
            val_unit_nid = self.add_val_unit(val_unit)
            self.add_edge_directed(order_by_nid, val_unit_nid)
        return order_by_nid
    
    def add_having_clause(self, conds):
        # assume where is not empty (checked outside function)
        having_nid = self.add_node('having')
        conds_nid = self.add_conditions(conds)
        self.add_edge_directed(having_nid, conds_nid)
        return having_nid

    def add_limit(self, limit):
        limit_nid = self.add_node('limit')
        self.add_edge_directed(limit_nid, self.add_node(str(limit)))
        return limit_nid
    
    def add_nested_sql(self, sql, text):
        nested_text_nid = self.add_node(text)
        sql_nid = self.add_sql(sql)
        self.add_edge_directed(nested_text_nid, sql_nid)
        return nested_text_nid
    
    def add_sql(self, sql):
        # add clauses
        root_nid = self.add_node('root')
        self.add_edge_directed(root_nid, self.add_select_clause(sql['select']))
        self.add_edge_directed(root_nid, self.add_from_clause(sql['from']))
        if len(sql['where']) > 0:
            self.add_edge_directed(root_nid, self.add_where_clause(sql['where']))
        if len(sql['groupBy']) > 0:
            self.add_edge_directed(root_nid, self.add_group_by_clause(sql['groupBy']))
        if len(sql['orderBy']) > 0:
            self.add_edge_directed(root_nid, self.add_order_by_clause(sql['orderBy']))
        if len(sql['having']) > 0:
            self.add_edge_directed(root_nid, self.add_having_clause(sql['having']))
        if sql['limit'] is not None:
            self.add_edge_directed(root_nid, self.add_limit(sql['limit']))
        if sql['intersect'] is not None:
            self.add_edge_directed(root_nid, self.add_nested_sql(sql['intersect'], 'intersect'))
        if sql['except'] is not None:
            self.add_edge_directed(root_nid, self.add_nested_sql(sql['except'], 'except'))
        if sql['union'] is not None:
            self.add_edge_directed(root_nid, self.add_nested_sql(sql['union'], 'union'))
        return root_nid
        
    def get_graph_info(self):
        info = {}
        info["text"] = self.text
        info['text_tokens'] = self.text_toks
        info['sql_original'] = self.query
        info['sql'] = self.query_toks
        
        info["g_ids"] = self.g_ids
        info['g_ids_features'] = self.g_id_feature
        info['g_adj'] = self.g_adj
        return info

# sql_paths is a list of input file names
def load_files(sql_paths,table_path,out_path):
    data_test = []
    table_test = []
    with open(table_path, "r+", encoding = 'utf-8') as f:
        table_test = json.loads(f.read())
    
    table_test_dict = {}
    for t in table_test:
        db_id = t['db_id']
        table_test_dict[db_id] = t
        orig_to_parsed_headers = {}
        for i, h in enumerate(t['column_names_original']):
            orig_col_name = h[1]
            col_name = t['column_names'][i][1]
            orig_to_parsed_headers[orig_col_name.lower()] = col_name.lower()
        for i, h in enumerate(t['table_names_original']):
            table_name = t['table_names'][i]
            orig_to_parsed_headers[h.lower()] = table_name.lower()
        t['to_parsed_headers'] = orig_to_parsed_headers
        
    f_test = open(out_path,"w", encoding = 'utf-8')
    for sql_path in sql_paths:
        with open(sql_path, "r+", encoding = 'utf-8') as f:
            data_test = json.loads(f.read())

        print("Loading %d queries from %d tables" % (len(data_test), len(table_test)))

        for i, example in enumerate(data_test):
            g = SpiderGraph(example, table_test_dict)
            info = g.get_graph_info()
            f_test.write(json.dumps(info)+"\n")
    f_test.close()

if __name__ == "__main__":
    load_files(["train_spider.json", "train_others.json"], "tables.json", "spider_train.json")
    load_files(["dev.json"], "tables.json", "spider_dev.json")

