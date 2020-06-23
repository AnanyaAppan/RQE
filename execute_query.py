import psycopg2
import psycopg2.extras
import pandas as pd

def main () :
    try:
        conn = psycopg2.connect("dbname=tpch2 host='localhost' user='ananya' password='*Rasika0507'")
        sql = """select s_acctbal, s_name, p_partkey, p_mfgr
from part, supplier, partsupp
where p_partkey = ps_partkey and s_suppkey = ps_suppkey"""
        dat = pd.read_sql_query(sql, conn)
        dat.to_csv('query.csv') 

    except psycopg2.Error as e:
        print(type(e))
        print(e)

if __name__ == "__main__" :
    main ()

    
