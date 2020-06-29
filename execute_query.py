import psycopg2
import psycopg2.extras
import pandas as pd

def main () :
    try:
        conn = psycopg2.connect("dbname=tpch host='localhost' user='ananya' password='*Rasika0507'")
        sql = """
            select
                c_name,
                n_name
            from
                customer,
                orders,
                lineitem,
                nation
            where
                c_custkey = o_custkey
                and l_orderkey = o_orderkey
                and c_nationkey = n_nationkey
            """
        dat = pd.read_sql_query(sql, conn)
        dat.to_csv('query.csv') 

    except psycopg2.Error as e:
        print(type(e))
        print(e)

if __name__ == "__main__" :
    main ()

    
