import psycopg2
import psycopg2.extras
import pandas as pd
import sys
from utils import *
import time

def main () :
    start = time.time()
    try:
        conn = psycopg2.connect("dbname=tpch2 host='localhost' user='ananya' password='*Rasika0507'")
        df = pd.read_csv("query.csv",header=0,index_col=0)
        tables = get_tables(conn)
        table_dict = get_table_dict(conn,tables)
        # pre_process(table_dict,conn)
        # print("--------------Pre processing over--------------%f"%(time.time()-start))
        cand_dict = get_c_and_lists(conn,table_dict,df)
        print(cand_dict)
        print("----Obtained CAND lists----%f"%(time.time()-start))
        # all_possib = get_all_possibilities(cand_dict,list(cand_dict.keys())) 
        # print(all_possib)
        # sys.exit()
        for depth in range(4):
            print("----trying for depth %d----%f"%(depth,time.time()-start))
            # for possib in all_possib:
            star_ctrs,tree_dict = gen_instance_trees(conn,cand_dict,depth)
            # print(tree_dict)
            # if(len(possib) != len(tree_dict.keys())) : continue
            valid = None
            merge = None
            theta = df.sample(n=min(5,len(df)))
            for _, row in theta.iterrows():
                ExploreInstanceTree(conn,tree_dict,row)
                print("-------Exploring Instance Tree-------%f"%(time.time()-start))
                star_ctrs_copy = [x for x in star_ctrs]
                for star in star_ctrs_copy :  
                    valid,tree_dict = updateStarCtrs(tree_dict,star,valid)
                    print("-------Update Star Centres-------%f"%(time.time()-start))
                    if valid[star] == {} :
                        if(star in star_ctrs) : star_ctrs.remove(star)
                for table in table_dict :
                    merge = get_merge_list(tree_dict,table,merge)
                print("-------Obtained merge list-------%f"%(time.time()-start))
                print(valid)
                print(merge)

            # print(star_ctrs)
            # print(valid)
            # for col in tree_dict:
            #     tree = tree_dict[col]
            #     draw_tree(tree)
            if(len(star_ctrs)==0): continue
            if(valid != None):
                S = get_starred_set(valid)
                merged_stars = []
                for s in S :
                    merged_stars.append(merge_stars(s,tree_dict))
                for merged_star in merged_stars :
                    initialize_tid_lists(merged_star,merge)
                    query = gen_lattice(merged_star,df,merge)
                    if(query != None):
                        print(query)
                        # post_process(table_dict,conn)
                        # print("--------------Post processing over--------------%f"%(time.time()-start))
                        sys.exit()
        # post_process(table_dict,conn)
        # print("--------------Post processing over--------------%f"%(time.time()-start))

    except psycopg2.Error as e:
        print(type(e))
        print(e)

if __name__ == "__main__" :
    main ()
    
