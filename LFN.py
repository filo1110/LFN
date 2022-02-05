import json
import urllib.request
from os import system
import time
import math

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import random
from random import randrange
import itertools
import ssl


def ComputeGraph():
    # Import data for airports
    airportJsonData = importAirportDataFromJson()

    # We select only the columns we are interested in
    airportDataFrame = pd.DataFrame.from_records(airportJsonData, columns=['city_code', 'country_code','name','code'])

    europeCountries = ["BE","BG","CZ","DK","DE","EE","IE","EL","ES","FR","HR","IT","CY","LV","LT","LU","HU","MT","NL","AT","PL","PT","RO","SI","SK","FI","SE"]

    europeAirportDataFrame = pd.DataFrame()

    # Use only Europe nodes
    for state in europeCountries:        
        airport = airportDataFrame.query('country_code == "'+ state +'"')
        europeAirportDataFrame = pd.concat([europeAirportDataFrame, airport])

    #print(europeAirportDataFrame)

    # To select the world graph, uncomment the next 2 lines and comment the previous for loop
    #airport = airportDataFrame
    #europeAirportDataFrame = pd.concat([europeAirportDataFrame, airport])
    
    europeAirportCodes = europeAirportDataFrame['code']

    # Import data for routes
    routesJsonData = importRoutesDataFromJson()

    # We select only the columns we are interested in
    europeRoutesDataFrame = pd.DataFrame.from_records(routesJsonData, columns=['departure_airport_iata', 'arrival_airport_iata'])

    # Additional column for the flight between two nodes
    europeRoutesDataFrame['flights'] = len(routesJsonData[0]["planes"])

    # filtered routes for origin and destination airports within EU if you don't select the entire world
    europeanRoutesDataFrame = europeRoutesDataFrame.loc[(europeRoutesDataFrame['departure_airport_iata'].isin(europeAirportCodes)) & (europeRoutesDataFrame['arrival_airport_iata'].isin(europeAirportCodes))]
    
    # calculate the count between two airports in any direction
    europeanRoutes = pd.DataFrame(europeanRoutesDataFrame.groupby(['departure_airport_iata', 'arrival_airport_iata']).size().reset_index(name='counts'))
    
    
    # Use only the routes with more than X connections (to use it uncomment the line below)
    #europeanRoutes = europeanRoutes[europeanRoutes['counts'] > 5]

    # Prepare the graph
    initialGraph = nx.from_pandas_edgelist(europeanRoutes, source='departure_airport_iata', target='arrival_airport_iata')

    allConnectedComponents = sorted(nx.connected_components(initialGraph), key=len, reverse=True)

    # Select only the major connected component
    finalGraph = initialGraph.subgraph(allConnectedComponents[0])
    
    # Draw the graph
    GraphDrawing(europeanRoutes, finalGraph)

    print('Number of nodes= ' + str(finalGraph.number_of_nodes()))
    print("\n")
    
    print('#######################################')
    # Compute centralities
    print("\n")
    print('Computing centralities...')
    print("\n")
    print("---------------------------------------")
    #degree_dict = DegreeCentrality(finalGraph)
    print("---------------------------------------")
    #closeness_dict = ClosenessCentrality(finalGraph)
    print("---------------------------------------")
    #ApproximateClosenessCentrality(finalGraph)
    print("---------------------------------------")
    #betweenness_dict = BetweennessCentrality(finalGraph)
    #topSubGraph(finalGraph,betweenness_dict,10)
    print("---------------------------------------")
    #ApproximateBetweennessCentrality(finalGraph)
    print("---------------------------------------")
    print("\n")
    print('#######################################')
    # Count triangles
    print("\n")
    print('Counting graphlets...')
    print("\n")
    print("---------------------------------------")
    countTriangles(finalGraph)
    print("---------------------------------------")
    LCC(finalGraph)
    print("---------------------------------------")
    approximateLCC(finalGraph,10)
    LCCtest(finalGraph)
    print("---------------------------------------")
    print("\n")
    print('#######################################')
    

def GraphDrawing(routes, graph):
    plt.figure(figsize=(50, 50))

    # Create a layout for our nodes
    layout = nx.spring_layout(graph, iterations=50)
    
    # 3. Draw the parts we want
    nx.draw_networkx_edges(graph, layout, edge_color='#AAAAAA')

    destinationNode = [node for node in graph.nodes() if node in routes.arrival_airport_iata.unique()]
    # The bigger the node, the higher its degree
    size = [graph.degree(node) * 80 for node in graph.nodes() if node in routes.arrival_airport_iata.unique()]
    nx.draw_networkx_nodes(graph, layout, nodelist=destinationNode, node_size=size, node_color='lightblue')

    originNode = [node for node in graph.nodes() if node in routes.departure_airport_iata.unique()]
    nx.draw_networkx_nodes(graph, layout, nodelist=originNode, node_size=100, node_color='#AAAAAA')

    selectedHighDegreeNodes = [node for node in graph.nodes() if node in routes.departure_airport_iata.unique() and graph.degree(node) > 1]
    nx.draw_networkx_nodes(graph, layout, nodelist=selectedHighDegreeNodes, node_size=100, node_color='#fc8d62')

    originNodesLables = dict(zip(originNode, originNode))
    nx.draw_networkx_labels(graph, layout, labels=originNodesLables)

    plt.axis('off')
    plt.title("Connections between Airports and Railway Stations in Europe")
    plt.show()


# Degree centrality  
def DegreeCentrality(graph):
    # Counting the amount of time needed
    startTime= time.time()

    # We use the exact algorithm in networkX
    degreeCentrality = nx.degree_centrality(graph)

    endTime = time.time()
    print("Degree centrality time: " + str(endTime-startTime))

    degreeCentralityDataFrame = pd.DataFrame(degreeCentrality.items())
    #print(degreeCentralityDataFrame)

    # Uncomment the line below if you want to select less centrality in the plot
    #degreeCentralityDataFrame = degreeCentralityDataFrame[degreeCentralityDataFrame[1] > 0.25]

    #Plotting the results
    plt.bar(degreeCentralityDataFrame[0], degreeCentralityDataFrame[1])
    plt.xlabel('Airports')
    plt.ylabel('Degree Centrality')
    plt.show()
    
    return degreeCentrality


# Closeness centrality
def ClosenessCentrality (graph):
    # Counting the amount of time needed
    startTime = time.time()

    # We use the exact algorithm in networkX
    closenessCentrality = nx.closeness_centrality(graph)
    
    endTime = time.time()
    print("Closeness centrality time: " + str(endTime-startTime))

    closenessCentralityDataFrame = pd.DataFrame(closenessCentrality.items())
    #print(closenessCentralityDataFrame)

    # Uncomment the line below if you want to select less centrality in the plot
    #closenessCentralityDataFrame = closenessCentralityDataFrame[closenessCentralityDataFrame[1] > 0.55]

    #Plotting the results
    plt.bar(closenessCentralityDataFrame[0], closenessCentralityDataFrame[1])
    plt.xlabel('Airports')
    plt.ylabel('Closeness Centrality')
    plt.show()
    
    return closenessCentrality
    

 # Approximate Closeness centrality   
def ApproximateClosenessCentrality(graph):
    # Values found with different experiments
    epsilon = 0.1
    sigma = 0.01
    nodesDictionary = []
    nodesList = []
    
    # Finding the lower bound k for which we have great approximation
    lowerLimitK = math.ceil((1/(2*pow(epsilon, 2)))*math.log((2*graph.number_of_nodes())/sigma,10)*pow(graph.number_of_nodes()/(graph.number_of_nodes()-1), 2))
    print("Selected k for ApproximateClosenessCentrality: " + str(lowerLimitK))    

    # Our implementation of the Eppstein-Wang algorithm
    for node in graph.nodes:
        nodesList.append(node)

    # Counting the amount of time needed
    startTime = time.time()

    nodesDictionary = dict.fromkeys(nodesList, 0)

    for i in range(lowerLimitK):
        randomNumber = randrange(len(nodesList))
        randomDIctionary = nx.algorithms.shortest_paths.unweighted.single_source_shortest_path_length(graph, nodesList[randomNumber])

        for nodeKey in nodesDictionary.keys():
            nodesDictionary[nodeKey] = nodesDictionary[nodeKey] + randomDIctionary[nodeKey]

    for nodeKey, nodeValue in nodesDictionary.items():
        nodesDictionary[nodeKey] = 1/((graph.number_of_nodes()*nodeValue)/(lowerLimitK*(graph.number_of_nodes()-1)))
    #Finish implementation of Eppstein-Wang algorithm

    endTime = time.time()
    print("Approximate Closeness centrality time: " + str(endTime-startTime))

    approxClosenessDataFrame = pd.DataFrame(nodesDictionary.items())
    #print(approxClosenessDataFrame)
    
    # Uncomment the line below if you want to select less centrality in the plot
    # approxClosenessDataFrame = approxClosenessDataFrame[approxClosenessDataFrame[1] > 0.55]

    #Plotting the results    
    plt.bar(approxClosenessDataFrame[0], approxClosenessDataFrame[1])
    plt.xlabel('Airports')
    plt.ylabel('Approximated Closeness Centrality')
    plt.show()


# Betweenness centrality
def BetweennessCentrality(graph):    
    # Counting the amount of time needed    
    startTime = time.time()

    # We use the exact algorithm in networkX
    betweennessCentrality = nx.betweenness_centrality(graph)

    endTime = time.time()
    print("Betweenness centrality time: " + str(endTime-startTime))

    betweennessCentralityDataFrame = pd.DataFrame(betweennessCentrality.items())
    #print(betweennessCentralityDataFrame)

    # Uncomment the line below if you want to select less centrality in the plot
    # betweennessCentralityDataFrame = betweennessCentralityDataFrame[betweennessCentralityDataFrame[1] > 0.05]

    #Plotting the results
    plt.bar(betweennessCentralityDataFrame[0], betweennessCentralityDataFrame[1])
    plt.xlabel('Airports')
    plt.ylabel('Betweenness Centrality')
    plt.show()
    
    return betweennessCentrality


# Approximate Betweenness centrality  
def ApproximateBetweennessCentrality(graph):    
    # Compute the needed parameters
    diameter = nx.algorithms.distance_measures.diameter(graph)

    # Values found with different experiments
    epsilon = 0.2
    sigma = 0.01

    # Finding the lower bound k for which we have great approximation
    lowerLimitK = math.ceil((2/pow(epsilon, 2))*(int(math.log(diameter-2,2))+math.log(1/sigma)))
    print("Selected k for ApproximateBetweennessCentrality: " + str(lowerLimitK))    

    # Counting the amount of time needed
    startTime = time.time()

    # We use the approximate algorithm in networkX using the k that is the lower bound for which we have great approximation
    approximateBetweenness = nx.betweenness_centrality(graph, lowerLimitK)

    endTime = time.time()
    print("Betweenness centrality time: " + str(endTime-startTime))

    approximateBetweennessDataFrame = pd.DataFrame(approximateBetweenness.items())
    #print(approximateBetweennessDataFrame)

    # Uncomment the line below if you want to select less centrality in the plot
    # approximateBetweennessDataFrame = approximateBetweennessDataFrame[approximateBetweennessDataFrame[1] > 0.05]

    #Plotting the results
    plt.bar(approximateBetweennessDataFrame[0], approximateBetweennessDataFrame[1])
    plt.xlabel('Airports')
    plt.ylabel('Approximated Betweenness Centrality')
    plt.show()


def topSubGraph(graph, centrality, n):
    sorted_tuples = sorted(centrality.items(), key=lambda item: item[1])
    centralitySorted = {k: v for k, v in sorted_tuples}
    first_n = list(centralitySorted.items())[len(centralitySorted)-n:]
    list_nodes = []
    newGraph = nx.Graph()
    for k1,v1 in first_n:
        for k2,v2 in first_n:
            nodes_in_path = nx.shortest_path(graph, k1, k2)
            for i in range(len(nodes_in_path)-1):
                newGraph.add_node(nodes_in_path[i])
                if(i!=len(nodes_in_path)):
                    newGraph.add_edge(nodes_in_path[i],nodes_in_path[i+1])
    nx.draw(newGraph, with_labels = True)
    plt.show()
    plt.savefig("filename.png", format="PNG")

    
def countTriangles(graph):
    startTime = time.time()
    dict = nx.triangles(graph)
    endTime = time.time()
    sum = 0
    for key, value in dict.items():
        sum = sum + value
    sum = sum/3
    print("Number of triangles with networkX: " + str(sum))
    print("Computation time: " + str(endTime-startTime))
    
    
def LCC(graph):
    startTime = time.time()
    dict = nx.clustering(graph)
    endTime = time.time()
    print("LCC for each node with networkX: " + str(dict))
    print("Computation time: " + str(endTime-startTime))


def LCCtest(graph):
    startTime = time.time()
    lcc_dict = dict.fromkeys(nx.nodes(graph), 0)
    for node in nx.nodes(graph):
        neighbours=[n for n in nx.neighbors(graph,node)]
        n_neighbors=len(neighbours)
        n_links=0
        if n_neighbors>1:
            for node1 in neighbours:
                for node2 in neighbours:
                    if graph.has_edge(node1,node2):
                        n_links+=1
            n_links/=2 #because n_links is calculated twice
            clustering_coefficient=n_links/(0.5*n_neighbors*(n_neighbors-1))
            lcc_dict[node] = clustering_coefficient
    endTime = time.time()
    print("Approximated LCC for each node test: " + str(lcc_dict))
    print("Computation time: " + str(endTime-startTime))
    
    
def approximateLCC(graph,k):
    nodes_list = nx.nodes(graph)
    n_nodes = len(nodes_list)
    edges_list = nx.edges(graph)
    lcc_dict = dict.fromkeys(nodes_list, 0)
    nodes_dict = dict.fromkeys(nodes_list, 0)
    edges_dict = dict.fromkeys(edges_list, 0)
    startTime = time.time()
    for i in range(k):
        nodes_permutation = list(nodes_list)
        random.shuffle(nodes_permutation)
        for v in nodes_permutation:
            minv = n_nodes
            for u in graph.neighbors(v):
                tmp_min = nodes_permutation.index(u)
                if(tmp_min < minv):
                    minv = tmp_min
            nodes_dict[v] = minv
        for e in edges_list:
            if(nodes_dict[e[0]] == nodes_dict[e[1]]):
                edges_dict[e] = edges_dict[e] + 1
    for v in lcc_dict.keys():
        sum = 0
        degv = graph.degree(v)
        for u in graph.neighbors(v):
            degu = graph.degree(u)
            if((v,u) in edges_dict.keys()):
                sum = sum + (edges_dict[(v,u)]/(edges_dict[(v,u)]+k))*(degu+degv)
            else:
                sum = sum + (edges_dict[(u,v)]/(edges_dict[(u,v)]+k))*(degu+degv)
        if(degv != 1):
            lcc_dict[v] = sum*(1/(degv*(degv-1)))
    endTime = time.time()
    print("Approximate LCC for each node: " + str(lcc_dict))
    print("Computation time: " + str(endTime-startTime))


def importRoutesDataFromJson():
    url = "http://api.travelpayouts.com/data/routes.json"

    with urllib.request.urlopen(url) as url:
        data = json.loads(url.read().decode("utf-8"))

    return data


def importAirportDataFromJson():
    url = 'https://api.travelpayouts.com/data/en/airports.json'
    ctx = ssl._create_unverified_context()
    with urllib.request.urlopen(url, context=ctx) as url:
        airport_json = json.loads(url.read().decode("utf-8"))

    return airport_json



if __name__ == '__main__':
    ComputeGraph()
