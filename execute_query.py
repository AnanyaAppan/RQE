import psycopg2
import psycopg2.extras
import pandas as pd

def main () :
    try:
        conn = psycopg2.connect("dbname=tpch host='localhost' user='ananya' password='*Rasika0507'")
        # sql = """   SELECT PS.ps_supplycost, L.l_shipdate, O.o_orderdate 
        #             FROM partsupp as PS, partsupp as PS1, Part as P, supplier as S, lineitem as L, lineitem as L1, orders as O 
        #             WHERE PS1.ps_suppkey = S.s_suppkey and S.s_suppkey = PS.ps_suppkey and PS1.ps_partkey = P.p_partkey and P.p_partkey = L.l_partkey and PS1.ps_partkey = L1.l_partkey and PS1.ps_suppkey = L1.l_suppkey and L1.l_orderkey = O.o_orderkey"""
        sql = """
        select l_orderkey, o_orderdate, o_shippriority
        from customer, orders, lineitem
        where c_custkey = o_custkey and l_orderkey = o_orderkey
        """
        dat = pd.read_sql_query(sql, conn)
        dat.to_csv('query4.csv') 

    except psycopg2.Error as e:
        print(type(e))
        print(e)

if __name__ == "__main__" :
    main ()
    
