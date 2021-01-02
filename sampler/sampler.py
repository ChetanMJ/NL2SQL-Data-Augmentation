from lib.dbengine import DBEngine
from lib.table import Table
#from lib.query import Query
import json
import jsonlines

#create DataBase Engine
DB=DBEngine("data/train.db")

#load table
tables=[json.loads(x) for x in open("data/train.tables.jsonl")]
print(len(tables))
#Create Table 
table=Table(tables[0]['id'], tables[0]['header'], tables[0]['types'], tables[0]['rows'])
print(table)

#sample
print(table.generate_queries(DB.conn,n=5, max_tries=5, lower=True))
queries = table.generate_queries(DB.conn,n=5, max_tries=5, lower=True)
print(str(queries[0][0]))
def augment_data():
    new_train_path = "data/augmented_train.jsonl"
    error_cnt = 0
    with open(new_train_path,'w') as f:
        for i in range(len(tables)):
            table_new = Table(tables[i]['id'], tables[i]['header'], tables[i]['types'], tables[i]['rows'])
            try:
                queries = table_new.generate_queries(DB.conn,n=10, max_tries=100, lower=True)
                for query in queries:
                    #print(query[1])
                    info = {}
                    sql = {}
                    sql['conds'] = query[0].conditions
                    #info['ordered'] = query[0].ordered
                    sql['sel'] = query[0].sel_index
                    sql['agg'] = query[0].agg_index
                    sql['exec_answer'] = query[1]
                    info['sql'] = sql
                    info['phase'] = 2 #default
                    info['table_id'] = tables[i]['id']
                    info['question'] = ''
                    f.write(json.dumps(info)+"\n")
            except:
                error_cnt += 1
    print(error_cnt)
augment_data()
