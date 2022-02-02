import json
import urllib.request
from os import system
import time
import math

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from random import randrange
import itertools


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

    europeAirportCodes = europeAirportDataFrame['code']
    #print(europeAirportDataFrame)

    # To select the world graph, uncomment the next 3 lines and comment the previous for loop
    #airport = airportDataFrame
    #europeAirportDataFrame = pd.concat([europeAirportDataFrame, airport])
    #europeAirportCodes = europeAirportDataFrame['code']

    # Import data for routes
    routesJsonData = importRoutesDataFromJson()

    # We select only the columns we are interested in
    europeRoutesDataFrame = pd.DataFrame.from_records(routesJsonData, columns=['departure_airport_iata', 'arrival_airport_iata'])

    # Additional column for the flight between two nodes
    europeRoutesDataFrame['flights'] = len(routesJsonData[0]["planes"])

    # filtered routes for origin and destination airports within EU if you don't select the entire world
    eruopeanRoutesDataFrame = europeRoutesDataFrame.loc[(europeRoutesDataFrame['departure_airport_iata'].isin(europeAirportCodes)) & (europeRoutesDataFrame['arrival_airport_iata'].isin(europeAirportCodes))]
    
    # calculate the count between two airports in any direction
    europeanRoutes = pd.DataFrame(eruopeanRoutesDataFrame.groupby(['departure_airport_iata', 'arrival_airport_iata']).size().reset_index(name='counts'))
    
    
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
    DegreeCentrality(finalGraph)
    print("---------------------------------------")
    ClosenessCentrality(finalGraph)
    print("---------------------------------------")
    ApproximateClosenessCentrality(finalGraph)
    print("---------------------------------------")
    BetweennessCentrality(finalGraph)
    print("---------------------------------------")
    ApproximateBetweennessCentrality(finalGraph)
    print("---------------------------------------")
    print("\n")
    print('#######################################')
    
    findGraphlets(finalGraph)
    

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


 # Approximate Closeness centrality   
def ApproximateClosenessCentrality ( graph):
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


def findGraphlets(graph):
    target = nx.Graph()
    target.add_edge(1,2)
    target.add_edge(2,3)
    target.add_edge(3,1)

    count_triangles = 0
    for sub_nodes in itertools.combinations(graph.nodes(),len(target.nodes())):
        subg = graph.subgraph(sub_nodes)
        if nx.is_connected(subg) and nx.is_isomorphic(subg, target):
            count_triangles = count_triangles + 1
            print(subg.edges())
    print(count_triangles)


def importRoutesDataFromJson():
    url = "http://api.travelpayouts.com/data/routes.json"

    with urllib.request.urlopen(url) as url:
        data = json.loads(url.read().decode("utf-8"))

    return data


def importAirportDataFromJson():
    url = 'https://api.travelpayouts.com/data/en/airports.json'

    with urllib.request.urlopen(url) as url:
        airport_json = json.loads(url.read().decode("utf-8"))

    return airport_json


if __name__ == '__main__':
    ComputeGraph()
