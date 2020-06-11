import psycopg2
import psycopg2.extras
from datetime import date
from decimal import *
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import sys

def pre_process(tables,connection) :
    cursor = connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    for table in tables :
        cursor.execute("""
                    ALTER TABLE %s
                    ADD COLUMN TID SERIAL
                    """%table)
    cursor.close
    connection.commit()

def get_tables(connection):
    """
    Create and return a list of dictionaries with the
    schemas and names of tables in the database
    connected to by the connection argument.
    """

    cursor = connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    cursor.execute("""SELECT table_schema, table_name
                      FROM information_schema.tables
                      WHERE table_schema != 'pg_catalog'
                      AND table_schema != 'information_schema'
                      AND table_type='BASE TABLE'
                      ORDER BY table_schema, table_name""")

    tables = cursor.fetchall()

    cursor.close()

    return tables

def get_columns(connection, table_schema, table_name):

    """
    Creates and returns a list of dictionaries for the specified
    schema.table in the database connected to.
    """

    where_dict = {"table_schema": table_schema, "table_name": table_name}
    cursor = connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cursor.execute("""SELECT column_name, ordinal_position, is_nullable, data_type, character_maximum_length
                      FROM information_schema.columns
                      WHERE table_schema = %(table_schema)s
                      AND table_name   = %(table_name)s
                      ORDER BY ordinal_position""",
                      where_dict)
    columns = cursor.fetchall()
    cursor.close()
    col_dict = {}
    for col in columns:
        col_dict[col["column_name"]] = col
    return col_dict

def get_primary_keys(connection) :
    cursor = connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cursor.execute("""
                    select kcu.table_name,kcu.column_name as key_column
                    from information_schema.table_constraints tco
                    join information_schema.key_column_usage kcu 
                        on kcu.constraint_name = tco.constraint_name
                        and kcu.constraint_schema = tco.constraint_schema
                        and kcu.constraint_name = tco.constraint_name
                    where tco.constraint_type = 'PRIMARY KEY'
                    order by kcu.ordinal_position;
                        """)
    data = cursor.fetchall()
    cursor.close()
    ret = {}
    for row in data :
        if row['table_name'] not in ret :
            ret[row['table_name']] = [row['key_column']]
        else :
            ret[row['table_name']].append(row['key_column'])
    for table in ret :
        ret[table].sort()
    return ret

def get_foreign_keys(connection) :
    cursor = connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cursor.execute("""
                    select kcu.table_schema || '.' ||kcu.table_name as foreign_table,
                    rel_tco.table_schema || '.' || rel_tco.table_name as primary_table,
                    string_agg(kcu.column_name, ', ') as fk_columns,
                    kcu.constraint_name
                    from information_schema.table_constraints tco
                    join information_schema.key_column_usage kcu
                            on tco.constraint_schema = kcu.constraint_schema
                            and tco.constraint_name = kcu.constraint_name
                    join information_schema.referential_constraints rco
                            on tco.constraint_schema = rco.constraint_schema
                            and tco.constraint_name = rco.constraint_name
                    join information_schema.table_constraints rel_tco
                            on rco.unique_constraint_schema = rel_tco.constraint_schema
                            and rco.unique_constraint_name = rel_tco.constraint_name
                    where tco.constraint_type = 'FOREIGN KEY'
                    group by kcu.table_schema,
                            kcu.table_name,
                            rel_tco.table_name,
                            rel_tco.table_schema,
                            kcu.constraint_name
                    order by kcu.table_schema,
                            kcu.table_name;
    """)
    columns = cursor.fetchall()
    cursor.close()
    for col in columns :
        col['fk_columns'] = col['fk_columns'].split(',')
        col['fk_columns'].sort()
        col['fk_columns'] = ','.join(col['fk_columns']).strip()
    return columns

def get_col_values (connection,col_name,table_name) :
    """
    Gets values of a clomun, goven table and column name
    """
    cursor = connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cursor.execute("""SELECT %s
                      FROM %s
                      """ % (col_name,table_name))
    vals = cursor.fetchall()
    cursor.close()
    ret_vals = []
    for val in vals :
        if type(val[col_name]) == str :
            ret_vals.append(val[col_name])
        elif type(val[col_name]) == date :
            ret_vals.append(str(val[col_name]))
        else :
            ret_vals.append(eval(str(val[col_name])))
    return ret_vals

def get_table_dict(connection, tables):
    """
    Creates and returns a dictionary with the table names
    as keys, and the dictionary returned by get_columns as 
    the corresponding values
    """
    d = {}
    for row in tables :
        d[row["table_name"]] = get_columns(connection, row["table_schema"], row["table_name"])
    return d

def get_c_and_lists(connection,table_dict,out) :
    
    distinct_col_values = {}
    for table in table_dict :
            for col in table_dict[table]:
                distinct_col_values[table + "." + col] = set(get_col_values(connection,col,table))

    ret_dict = {}
    for col_out in out :
        matching_cols = []
        for col in distinct_col_values:
            if set(out[col_out]).issubset(distinct_col_values[col]) :
                matching_cols.append(col)
        ret_dict[col_out] = matching_cols

    return ret_dict

def get_foreign_key_dict(primary_keys,foreign_keys) :
    ret_dict = {}
    for row in foreign_keys :
        table = row['foreign_table'].split(".")[1]
        primary_table = row['primary_table'].split(".")[1]
        foreign_key_cols = row['fk_columns'].split(',')
        for i in range(len(foreign_key_cols)):
            d = {}
            d["foreign_col"] = foreign_key_cols[i]
            d["primary_col"] = primary_keys[primary_table][i]
            d["primary_table"] = primary_table
            if table not in ret_dict:
                ret_dict[table] = [d]
            else :
                ret_dict[table].append(d)
    return ret_dict

def get_primary_key_dict(foregin_key_dict) :
    ret_dict = {}
    for table in foregin_key_dict :
        for constraint in foregin_key_dict[table] :
            primary_table = constraint["primary_table"]
            d = {}
            d["foreign_table"] = table
            d["foreign_col"] = constraint["foreign_col"]
            d["primary_col"] = constraint["primary_col"]
            if primary_table not in ret_dict:
                ret_dict[primary_table] = [d]
            else :
                ret_dict[primary_table].append(d)
    return ret_dict

def get_instance_tree(table_count,col,foreign_key_dict,primary_key_dict,depth) :
    table_name = col.split('.')[0]
    col_name = col.split('.')[1]
    G = nx.Graph()
    attr = {}
    if table_name + '_' + str(table_count[table_name]) not in attr :
        attr[table_name + '_' + str(table_count[table_name])] = {}
    G.add_node(table_name + '_' + str(table_count[table_name]))
    attr[table_name + '_' + str(table_count[table_name])]['col'] = [col_name]
    attr[table_name + '_' + str(table_count[table_name])]['star'] = 1
    queue = []
    queue.append(table_name + '_' + str(table_count[table_name]))
    table_count[table_name] += 1
    for _ in range(depth) :
        new_queue = []
        while(queue != []) :
            head = queue.pop(0)
            table = head.split('_')[0]
            try :
                for key in foreign_key_dict[table] :
                        table_name = key['primary_table']
                        G.add_edge(head,table_name + '_' + str(table_count[table_name]))
                        G[head][table_name + '_' + str(table_count[table_name])]["join"] = [table + "." + key['foreign_col']
                                                                                    ,table_name + "." + key['primary_col']]
                        if table_name + '_' + str(table_count[table_name]) not in attr :
                            attr[table_name + '_' + str(table_count[table_name])] = {}
                        attr[table_name + '_' + str(table_count[table_name])]['col'] = None
                        attr[table_name + '_' + str(table_count[table_name])]['star'] = 1
                        new_queue.append(table_name + '_' + str(table_count[table_name]))
                        table_count[table_name] += 1
            except :
                pass

            try :
                for key in primary_key_dict[table] : 
                        table_name = key['foreign_table']
                        G.add_edge(head,table_name + '_' + str(table_count[table_name]))
                        G[head][table_name + '_' + str(table_count[table_name])]["join"] = [table + "." + key['primary_col']
                                                                                        ,table_name + "." + key['foreign_col']]
                        if table_name + '_' + str(table_count[table_name]) not in attr :
                            attr[table_name + '_' + str(table_count[table_name])] = {}
                        attr[table_name + '_' + str(table_count[table_name])]['col'] = None
                        attr[table_name + '_' + str(table_count[table_name])]['star'] = 1
                        new_queue.append(table_name + '_' + str(table_count[table_name]))
                        table_count[table_name] += 1
            except : 
                pass
        queue = new_queue
    nx.set_node_attributes(G,attr)
    return table_count,G

def bottom_up_prune(tree,root,prev_node,is_star) :
    sub_trees = [node for node in tree[root]]
    for node in sub_trees :
        if(node != prev_node): tree = bottom_up_prune(tree,node,root,is_star)
    if (list(tree[root])==[prev_node] or len(list(tree[root]))==0 and root != list(tree.nodes())[0]) :
        if is_star[root] == 0 : 
            # print("removed ",root)
            tree.remove_node(root)
    return tree
    

def gen_instance_trees(connection,possib) : 

    tables = get_tables(connection)
    table_dict = get_table_dict(connection,tables)
    foreign_keys = get_foreign_keys(connection)
    primary_keys = get_primary_keys(connection)
    foreign_key_dict = get_foreign_key_dict(primary_keys,foreign_keys)
    primary_key_dict = get_primary_key_dict(foreign_key_dict)
    depth = 1
    tree_dict = {}
    table_count = {}
    for table in table_dict :
        table_count[table] = 0

    for col in possib :
        table_count,tree = get_instance_tree(table_count,col,foreign_key_dict,primary_key_dict,depth)
        tree_dict[col] = tree
        # nx.draw(tree,with_labels=True)
        # plt.show()
    
    star_ctrs = set([table for table in table_dict])
    for col_out in tree_dict :
        tree = tree_dict[col_out]
        star_ctrs = star_ctrs.intersection(set([table.split('_')[0] for table in tree.nodes()]))

    for col_out in tree_dict :
        tree = tree_dict[col_out]
        root = list(tree.nodes())[0]
        attr = {}
        for node in list(tree.nodes()):
            attr[node] = {}
            if((node.split('_')[0] in star_ctrs)): attr[node]["star"] = 1
            else: attr[node]["star"] = 0
        nx.set_node_attributes(tree,attr)
    
    for col_out in tree_dict : 
        is_star = nx.get_node_attributes(tree_dict[col_out],'star')
        pruned_tree = bottom_up_prune(tree_dict[col_out],list(tree_dict[col_out].nodes())[0],None,is_star)
        tree_dict[col_out] = pruned_tree
        # nx.draw(pruned_tree,with_labels=True)
        # plt.show()
    
    return star_ctrs,tree_dict

def get_tid_util(tree,prev_node,cur_node,attr,conn) :
    if(len(attr[prev_node]['tid'])==0) : return attr,tree
    cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur_table = cur_node.split('_')[0]
    prev_table = prev_node.split('_')[0]
    query = """SELECT DISTINCT %s
                FROM %s,%s
                WHERE %s = %s
                AND %s IN %s
            """%(cur_table+".TID",cur_table,prev_table,tree[prev_node][cur_node]["join"][0],
                tree[prev_node][cur_node]["join"][1],prev_table + ".TID" ,
                "(" + ",".join(attr[prev_node]['tid']).strip(",") + ")")
    cursor.execute(query)
    tables = cursor.fetchall()
    tables = [str(x['tid']) for x in tables]
    cursor.close()
    attr[cur_node]['tid'] = tables
    for node in tree[cur_node] :
        if(node != prev_node): attr,tree = get_tid_util(tree,cur_node,node,attr,conn)
    return attr,tree


def get_tid_lists(tree,conn,val) :

    if(type(val)==date or type(val)==str):
        val = "'" + str(val) + "'"
    root = list(tree.nodes())[0]
    cols = nx.get_node_attributes(tree,"col")
    attr = {}
    for node in list(tree.nodes()):
        attr[node] = {}
        attr[node]["tid"] = []
    root_table = root.split('_')[0]
    cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cursor.execute("""SELECT DISTINCT TID
                      FROM %s
                      WHERE %s = %s"""%(root_table,cols[root][0],str(val)))
    tables = cursor.fetchall()
    tables = [str(x['tid']) for x in tables]
    cursor.close()
    attr[root]["tid"] += tables
    for node in tree[root] : 
        attr,tree = get_tid_util (tree,root,node,attr,conn)
    nx.set_node_attributes(tree,attr)
    return tree

def ExploreInstanceTree(conn,tree,val) :
    tree = get_tid_lists(tree,conn,val) 
    tree = del_empty_tid(tree) 
    # tids = nx.get_node_attributes(tree,'tid')
    # pos = nx.spring_layout(tree)
    # nx.draw(tree, pos)
    # nx.draw_networkx_labels(tree, pos, labels = tids)
    # plt.show()

def draw_tree(tree) :
    attr = nx.get_node_attributes(tree,'tid')
    pos = nx.spring_layout(tree)
    nx.draw(tree, pos,with_labels=True)
    # nx.draw_networkx_labels(tree, pos, labels = attr)
    plt.show()

def del_empty_tid(tree) :
    empty_nodes = [x for x,y in tree.nodes(data=True) if len(y['tid'])==0]
    tree.remove_nodes_from(empty_nodes)
    return tree

def get_all_possibilities(c_and_dict,keys) : 
    if len(keys)==0 : return [[]]
    ret = []
    for possib in get_all_possibilities(c_and_dict,keys[1:]):
        for col in c_and_dict[keys[0]]:
            ret.append([col]+possib)
    return ret

def get_valid(tree_dict,star,is_table_star):
    valid = {}
    valid[star] = {} # valid list for star
    for col in tree_dict:
        tree = tree_dict[col]
        tids = nx.get_node_attributes(tree,'tid')
        is_star = nx.get_node_attributes(tree,'star')
        for node in list(tree.nodes()) :
            if((node.split('_')[0]==star and is_star[node] and is_table_star) or (node.split('_')[0]==star and not is_table_star)):
                for tid in tids[node]:
                    if(tid not in valid[star]):
                        valid[star][tid] = {} # inverted list for star
                        valid[star][tid][col] = set([node])
                    else :
                        if(col not in valid[star][tid]):
                            valid[star][tid][col] = set([node])
                        else :
                            valid[star][tid][col].add(node)
    tid_keys = [k for k in valid[star].keys()]
    if is_table_star : 
        for tid in tid_keys :
            if len(valid[star][tid].keys()) != len(tree_dict.keys()):
                del valid[star][tid]
    return valid

def cross_tuple_prune(valid,prev_valid,star):
    if(prev_valid==None): return valid
    if(star not in prev_valid): prev_valid[star] = valid[star]
    prev_valid_tids = [k for k in prev_valid[star].keys()]
    for tid in prev_valid_tids :
        # if tid not in valid[star] : del prev_valid[star][tid]
        # else : 
        for col in prev_valid[star][tid]:
            if tid in valid[star] : 
                if col in valid[star][tid] : 
                    prev_valid[star][tid][col] = prev_valid[star][tid][col].intersection(valid[star][tid][col])
    return prev_valid    

def updateStarCtrs(tree_dict,star,prev_valid):
    # l = []
    # for col in tree_dict:
    #     tree = tree_dict[col]
    #     tid_lists = [y['tid'] for x,y in tree.nodes(data=True) if x.split('_')[0]==star]
    #     tid_union = set()
    #     for tid_list in tid_lists :
    #         tid_union = tid_union.union(set(tid_list))
    #     l.append(tid_union)
    # tid_intersection = l[0]
    # for tids in l[1:]:
    #     tid_intersection = tid_intersection.intersection(tids)
    
    # for col in tree_dict:
    #     tree = tree_dict[col]
    #     stars_to_be_removed = [x for x,y in tree.nodes(data=True) 
    #                         if (tid_intersection.intersection(list(y['tid']))==set() and x.split('_')[0]==star)]
    #     is_star = nx.get_node_attributes(tree,'star')
    #     for star in stars_to_be_removed:
    #         is_star[star] = 0
    #     nx.set_node_attributes(tree,is_star,'star')
    #     tree_dict[col] = bottom_up_prune(tree,list(tree.nodes())[0],None,is_star)
    #     # draw_tree(tree)

    valid = get_valid(tree_dict,star,True)
    new_valid = cross_tuple_prune(valid,prev_valid,star)
    for col in tree_dict:
        tree = tree_dict[col]
        is_star = nx.get_node_attributes(tree,'star')
        star_nodes = [x for x,y in tree.nodes(data=True) 
                             if (y['star']==1 and x.split('_')[0]==star)]
        updated_candidates_for_star = set()
        for tid in new_valid[star] :
            for table in new_valid[star][tid][col]:
                updated_candidates_for_star.add(table)
        for star_node in star_nodes :
            if star_node not in updated_candidates_for_star and star_node.split('_')[0] == star:
                is_star[star_node] = 0
        nx.set_node_attributes(tree,is_star,'star')
        # draw_tree(tree)
        tree_dict[col] = bottom_up_prune(tree,list(tree.nodes())[0],None,is_star)
        # draw_tree(tree_dict[col])
    return new_valid,tree_dict

def get_starred_set(valid) :
    ret = []
    for star in valid : 
        for tid in valid[star] : 
            l = get_all_possibilities(valid[star][tid],list(valid[star][tid].keys()))
            for i in l:
                if i not in ret : ret.append(i)
    return ret

def substitute_node(G,old_node,new_node):
    G.add_node(new_node)
    for node in G[old_node] :
        G.add_edge(new_node,node)
        G[new_node][node]['join'] = G[old_node][node]['join']
    G.remove_node(old_node)

def merge_stars(star_list,tree_dict) :
    star = star_list[0].split('_')[0]
    table_nos = '_'.join([x.split('_')[1] for x in star_list])
    merged_star = star + '_' + table_nos
    merged_tree = nx.Graph()
    keys = list(tree_dict.keys())
    merged_tree_cols = {}
    for i in range(len(keys)):
        col = keys[i]
        tree = tree_dict[col]
        tree_cols = nx.get_node_attributes(tree,'col')
        root = list(tree.nodes())[0]
        merged_tree_cols[root] = tree_cols[root]
        star = star_list[i]
        if(star != root): 
            try :
                path = nx.Graph(tree.subgraph(list(nx.all_simple_paths(tree,root,star))[0]))
            except :
                print(root)
                print(star)
                draw_tree(tree)
        else : 
            path = nx.Graph(tree.subgraph(root))
            if(merged_star in merged_tree_cols): merged_tree_cols[merged_star] += tree_cols[root]
            else : merged_tree_cols[merged_star] = tree_cols[root]
        if(len(keys)>1): substitute_node(path,star,merged_star)
        merged_tree.add_nodes_from(path.nodes())
        merged_tree.add_edges_from(path.edges())
        for edge in path.edges() :
            node1 = edge[0]
            node2 = edge[1]
            merged_tree[node1][node2]['join'] = path[node1][node2]['join']

    nx.set_node_attributes(merged_tree,merged_tree_cols,'col')
    return merged_tree

def get_merge_list(tree_dict,table,prev_merge) :
    merge_list = get_valid(tree_dict,table,False)
    new_merge = cross_tuple_prune(merge_list,prev_merge,table)
    return new_merge

def initialize_tid_lists(tree, merge) :
    nodes = list(tree.nodes())
    tids = {}
    tables = set()
    for node in nodes : 
        tids[node] = set()
        table = node.split('_')[0]
        tables.add(table)
    for table in tables : 
        for tid in merge[table] :
            for col in merge[table][tid] :
                for candidate in merge[table][tid][col] :
                    if candidate in nodes:
                        tids[candidate].add(tid)
    nx.set_node_attributes(tree,tids,'tid')

def merge(graph,node1,node2) :
    no = '_'.join(node1.split('_')[1:]) + '_' + '_'.join(node2.split('_')[1:])
    new_node = node1.split('_')[0] + '_' + no
    for node in graph[node1] :
        graph.add_edge(node,new_node)
        graph[node][new_node]['join'] = graph[node][node1]['join']
    for node in graph[node2] :
        graph.add_edge(node,new_node)
        graph[node][new_node]['join'] = graph[node][node2]['join']
    tids = nx.get_node_attributes(graph,'tid')
    col = nx.get_node_attributes(graph,'col')
    if(node1 in col and node2 in col) :
        col[new_node] = col[node1] + col[node2]
    elif(node1 in col) :
        col[new_node] = col[node1]
    elif(node2 in col) :
        col[new_node] = col[node2]
    new_node_tid = tids[node1].intersection(tids[node2])
    tids[new_node] = new_node_tid
    nx.set_node_attributes(graph,tids,'tid')
    nx.set_node_attributes(graph,col,'col')
    graph.remove_node(node1)
    graph.remove_node(node2)

def execute_query(query) :
    conn = psycopg2.connect("dbname=tpch host='localhost' user='ananya' password='*Rasika0507'")
    dat = pd.read_sql_query(query, conn)
    return dat

def get_query_from_graph (graph) :
    nodes = list(graph.nodes())
    query = " SELECT "
    col = nx.get_node_attributes(graph,'col')
    projected_tables = col.keys()
    for projected_table in projected_tables :
        for x in set(col[projected_table]):
            query += projected_table + '.' + x + ", "
    query = query.strip(", ")
    query += " FROM "
    for node in nodes :
        query += node.split('_')[0] + " " + node + " , "
    query = query.strip(", ")
    if(len(list(graph.edges()))):
        query += " WHERE "
        for edge in list(graph.edges()) :
            node1 = edge[0]
            node2 = edge[1]
            join = graph[node1][node2]["join"]
            table_1 = join[0].split('.')[0]
            if(table_1 == node1.split('_')[0]) :
                query += node1 + "." + join[0].split('.')[1] + "=" + node2 + "." + join[1].split('.')[1] + " and "
            elif (table_1 == node2.split('_')[0]) :
                query += node2 + "." + join[0].split('.')[1] + "=" + node1 + "." + join[1].split('.')[1] + " and "      
        query = query.strip(" and ")
    # print(query)
    return query

def gen_lattice(graph,df) :

    try :
        query = get_query_from_graph(graph)
        res_df = execute_query(query)
        if df.sort_index(axis=1).equals(res_df.sort_index(axis=1)) :
            return query
    except : pass

    gen_graphs = []
    tids = nx.get_node_attributes(graph,'tid')
    nodes = list(graph.nodes())
    for i in range(len(nodes)):
        node1 = nodes[i]
        for j in range(i+1 , len(nodes)):
            node2 = nodes[j]
            if node1.split('_')[0] == node2.split('_')[0] and tids[node1].intersection(tids[node2]) != set():
                new_graph = nx.Graph(graph)
                merge(new_graph,node1,node2)
                is_isomorphic = False
                for g in gen_graphs :
                    if nx.is_isomorphic(g,new_graph):
                        is_isomorphic = True
                        break
                if not is_isomorphic :
                    gen_graphs.append(new_graph)

    for ele in gen_graphs :
        ret = gen_lattice(ele,df)
        if(ret != None) :
            return ret
    return None














        
