import itertools
import json
import urllib.request
from collections import OrderedDict
from os import system

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


def analyse_graph():

    # get airport data dump
    airport_json = get_airport_data()
    # prepare a subset of the data with limited columns
    airport_df = pd.DataFrame.from_records(airport_json, columns=['city_code', 'country_code','name','code'])
    # print(json.dumps(airport_json, indent=4, ))
    # filter dataframe to use US airports only
    europe = ["BE","BG","CZ","DK","DE","EE","IE","EL","ES","FR","HR","IT","CY","LV","LT","LU","HU","MT","NL","AT","PL","PT","RO","SI","SK","FI","SE"]
    airport_us = pd.DataFrame()

    for state in europe:
        # Filter dataframe to use USA airports only
        airport = airport_df.query('country_code == "'+state+'"')
        airport_us = pd.concat([airport_us, airport])
        # get a ist of indexes with US airports for filtering of route data
        airport_us_in = airport_us['code']

    # get routes data dump
    routes_df = get_routes_data()
    # print(json.dumps(routes_df, indent=4, ))
    # prepare seubset of data with limited columns
    routes_us = pd.DataFrame.from_records(routes_df, columns=['departure_airport_iata', 'arrival_airport_iata'])
    # add a column to count flights between two airports
    routes_us['flights'] = len(routes_df[0]["planes"])
    # filtered routes for origin ad destination airports within US
    routes_us_f = routes_us.loc[(routes_us['departure_airport_iata'].isin(airport_us_in)) &
                          (routes_us['arrival_airport_iata'].isin(airport_us_in))]
    # calculate the count between two airports in any direction
    routes_us_g = pd.DataFrame(routes_us_f.groupby(['departure_airport_iata', 'arrival_airport_iata']).size().reset_index(name='counts'))
    # filter routes based on number connections more than 5
    #routes_us_g = routes_us_g[routes_us_g['counts'] > 3]
    # pass this dataframe to draw the network graph of connectivities
    draw_graph(routes_us_g)
    #calculate and show centralities(Closeness, Betweenness)
    centralities(routes_us_g)
    # motif, g = compute_significant_motif(routes_us_g) #get a subgraph of interest
    # find_motif(g, motif)


def get_routes_data():
    url = "http://api.travelpayouts.com/data/routes.json"
    with urllib.request.urlopen(url) as url:
        data = json.loads(url.read().decode("utf-8"))
    return data


def get_airport_data():
    url = 'https://api.travelpayouts.com/data/en/airports.json'
    with urllib.request.urlopen(url) as url:
        airport_json = json.loads(url.read().decode("utf-8"))
    return airport_json


def draw_graph(data):
    plt.figure(figsize=(50, 50))
    # 1. Create the graph
    g = nx.from_pandas_edgelist(data, source='departure_airport_iata', target='arrival_airport_iata')

    # 2. Create a layout for our nodes
    layout = nx.spring_layout(g, iterations=50)
    # 3. Draw the parts we want
    nx.draw_networkx_edges(g, layout, edge_color='#AAAAAA')

    dest = [node for node in g.nodes() if node in data.arrival_airport_iata.unique()]
    size = [g.degree(node) * 80 for node in g.nodes() if node in data.arrival_airport_iata.unique()]
    nx.draw_networkx_nodes(g, layout, nodelist=dest, node_size=size, node_color='lightblue')

    orig = [node for node in g.nodes() if node in data.departure_airport_iata.unique()]
    nx.draw_networkx_nodes(g, layout, nodelist=orig, node_size=100, node_color='#AAAAAA')

    high_degree_orig = [node for node in g.nodes() if node in data.departure_airport_iata.unique() and g.degree(node) > 1]
    nx.draw_networkx_nodes(g, layout, nodelist=high_degree_orig, node_size=100, node_color='#fc8d62')

    orig_dict = dict(zip(orig, orig))
    nx.draw_networkx_labels(g, layout, labels=orig_dict)

    # 4. Turn off the axis because I know you don't want it
    plt.axis('off')
    plt.title("Connections between Airports and Railway Stations in Europe")
    # 5. Tell matplotlib to show it
    plt.plot()
    plt.savefig('routes_us.jpg')


def centralities(data):
    # prepare graph object using dataset
    g = nx.from_pandas_edgelist(data, source='departure_airport_iata', target='arrival_airport_iata')
    # calculate degree centrality
    deg_cen = nx.degree_centrality(g)
    data_deg_cen = pd.DataFrame(deg_cen.items())
    plt.bar(data_deg_cen[0], data_deg_cen[1])
    plt.xlabel('Airports')
    plt.ylabel('Degree Centrality')
    plt.show()
    plt.savefig('degree.jpg')
    data_sorted = data_deg_cen.sort_values(by=[1], ascending=False)
    #print(data_sorted)
    data_sorted.to_csv("degree_sorted.cvs", sep=' ', index=False, header=False)

    # calculate closeness centrality
    cl_cen = nx.closeness_centrality(g)
    data_cl_cen = pd.DataFrame(cl_cen.items())
    # print(data)
    data_cl_cen = data_cl_cen[data_cl_cen[1] > 0.05]
    plt.bar(data_cl_cen[0], data_cl_cen[1])
    plt.xlabel('Airports')
    plt.ylabel('Closeness Centrality')
    plt.show()
    plt.savefig('closeness.jpg')
    data_sorted1 = data_cl_cen.sort_values(by=[1], ascending=False)
    data_sorted1.to_csv("closeness_sorted.cvs", sep=' ', index=False, header=False)

    # print(cl_cen)
    # calculate betweenness centrality
    bet_cen = nx.betweenness_centrality(g)
    data_bet_cen = pd.DataFrame(bet_cen.items())
    # print(data)
    data_bet_cen = data_bet_cen[data_bet_cen[1] > 0.05]
    plt.bar(data_bet_cen[0], data_bet_cen[1])
    plt.xlabel('Airports')
    plt.ylabel('Betweenness Centrality')
    plt.show()
    plt.savefig('betweenness.jpg')
    data_sorted2 = data_bet_cen.sort_values(by=[1], ascending=False)
    data_sorted2.to_csv("betweenness_sorted.cvs", sep=' ', index=False, header=False)

def compute_significant_motif(data):
    # prepare graph object using dataset
    #IDEA: scegliere motif utilizzando nodi con centralità maggiore
    g = nx.from_pandas_edgelist(data, source='departure_airport_iata', target='arrival_airport_iata')
    motif = 0
    return motif, g

def find_motif(g,motif):
    #find input motif
    motif_rank = max(max(motif.shortest_paths_dijkstra()))
    result = OrderedDict.fromkeys(g.vs['label'], 0)

    for node in g.vs:
        # Get relevant nodes around node of interest that might create the motif of interest
        nodes_to_expand = {node}
        for rank in range(motif_rank):
            nodes_expanded = nodes_to_expand
            for node_to_expand in nodes_to_expand:
                nodes_expanded = set.union(nodes_expanded, set(node_to_expand.neighbors()))
            nodes_to_expand = nodes_expanded

        # Look at all combinations
        for sub_nodes in itertools.combinations(nodes_to_expand, motif.vcount()):
            subg = g.subgraph(sub_nodes)
            if subg.is_connected() and subg.isomorphic(motif):
                result[node['label']] = result[node['label']]+1
    return result


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    analyse_graph()