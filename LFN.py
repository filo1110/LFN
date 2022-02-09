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

    # Compute centralities
    print('Computing centralities...')
    print("\n")
    degree_dict = DegreeCentrality(finalGraph)
    print("\n")

    closeness_dict = ClosenessCentrality(finalGraph)
    print("\n")

    ApproximateClosenessCentrality(finalGraph)
    print("\n")

    betweenness_dict = BetweennessCentrality(finalGraph)
    print("\n")
    
    ApproximateBetweennessCentrality(finalGraph)

    print('\n\nOther features...\n')

    print("\n")

    SubGraphWithTopNodes(finalGraph,betweenness_dict,10)
    print("\n")

    LocalClusteringCoefficent(finalGraph)
    print("\n")

    ApproximateLocalClusteringCoefficent(finalGraph,max([val for (node, val) in finalGraph.degree()]))
    

def GraphDrawing(routes, graph):
    plt.figure(figsize=(50, 50))

    # Create a layout for nodes
    layout = nx.spring_layout(graph, iterations=50)
    
    # Draw the parts we want
    nx.draw_networkx_edges(graph, layout, edge_color='#AAAAAA')

    destinationNode = [node for node in graph.nodes() if node in routes.arrival_airport_iata.unique()]
    # The bigger the node, the higher its degree
    size = [graph.degree(node) * 80 for node in graph.nodes() if node in routes.arrival_airport_iata.unique()]
    nx.draw_networkx_nodes(graph, layout, nodelist=destinationNode, node_size=size, node_color='lightblue')

    # Use different colors for the High Degree Nodes
    originNode = [node for node in graph.nodes() if node in routes.departure_airport_iata.unique()]
    nx.draw_networkx_nodes(graph, layout, nodelist=originNode, node_size=100, node_color='#AAAAAA')

    selectedHighDegreeNodes = [node for node in graph.nodes() if node in routes.departure_airport_iata.unique() and graph.degree(node) > 1]
    nx.draw_networkx_nodes(graph, layout, nodelist=selectedHighDegreeNodes, node_size=100, node_color='#fc8d62')

    originNodesLables = dict(zip(originNode, originNode))
    nx.draw_networkx_labels(graph, layout, labels=originNodesLables)

    plt.axis('off')
    plt.title("Connections between Airports in Europe")
    plt.show()
    #plt.savefig("connections_graph.png", format="PNG")


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
    #degreeCentralityDataFrame = degreeCentralityDataFrame[degreeCentralityDataFrame[1] > 0.055]

    #Plotting the results
    plt.bar(degreeCentralityDataFrame[0], degreeCentralityDataFrame[1])
    plt.xlabel('Airports')
    plt.ylabel('Degree Centrality')
    plt.show()
    #plt.savefig("exact_degree.png", format="PNG")
    
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
    #closenessCentralityDataFrame = closenessCentralityDataFrame[closenessCentralityDataFrame[1] > 0.38]

    #Plotting the results
    plt.bar(closenessCentralityDataFrame[0], closenessCentralityDataFrame[1])
    plt.xlabel('Airports')
    plt.ylabel('Closeness Centrality')
    plt.show()
    #plt.savefig("exact_closeness.png", format="PNG")
    
    return closenessCentrality

    

 # Approximate Closeness centrality   
def ApproximateClosenessCentrality(graph):
    # Values found with different experiments
    epsilon = 0.1
    sigma = 0.01
    nodesDictionary = []
    nodesList = []
    n_nodes = graph.number_of_nodes()
    
    # Finding the lower bound k for which we have great approximation
    lowerLimitK = math.ceil((1/(2*pow(epsilon, 2)))*math.log((2*n_nodes)/sigma,10)*pow(n_nodes/(n_nodes-1), 2))
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
        nodesDictionary[nodeKey] = 1/((n_nodes*nodeValue)/(lowerLimitK*(n_nodes-1)))
    #Finish implementation of Eppstein-Wang algorithm

    endTime = time.time()
    print("Approximated Closeness centrality time: " + str(endTime-startTime))

    approxClosenessDataFrame = pd.DataFrame(nodesDictionary.items())
    #print(approxClosenessDataFrame)
    
    # Uncomment the line below if you want to select less centrality in the plot
    #approxClosenessDataFrame = approxClosenessDataFrame[approxClosenessDataFrame[1] > 0.38]

    #Plotting the results    
    plt.bar(approxClosenessDataFrame[0], approxClosenessDataFrame[1])
    plt.xlabel('Airports')
    plt.ylabel('Approximated Closeness Centrality')
    plt.show()
    #plt.savefig("appr_closeness.png", format="PNG")



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
    #betweennessCentralityDataFrame = betweennessCentralityDataFrame[betweennessCentralityDataFrame[1] > 0.04]

    #Plotting the results
    plt.bar(betweennessCentralityDataFrame[0], betweennessCentralityDataFrame[1])
    plt.xlabel('Airports')
    plt.ylabel('Betweenness Centrality')
    plt.show()
    #plt.savefig("exact_betweenness.png", format="PNG")
    
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
    print("Approximated Betweenness centrality time: " + str(endTime-startTime))

    approximateBetweennessDataFrame = pd.DataFrame(approximateBetweenness.items())
    #print(approximateBetweennessDataFrame)

    # Uncomment the line below if you want to select less centrality in the plot
    #approximateBetweennessDataFrame = approximateBetweennessDataFrame[approximateBetweennessDataFrame[1] > 0.04]

    #Plotting the results
    plt.bar(approximateBetweennessDataFrame[0], approximateBetweennessDataFrame[1])
    plt.xlabel('Airports')
    plt.ylabel('Approximated Betweenness Centrality')
    plt.show()
    #plt.savefig("appr_betweennees.png", format="PNG")



# Local Clustering Coefficent
def LocalClusteringCoefficent(graph):
    # Counting the amount of time needed
    startTime = time.time()
    
    localClusteringDictionary = dict.fromkeys(nx.nodes(graph), 0)

    # Start Implementation of the exact algorithm for LCC
    for node in nx.nodes(graph):
        neighbours=[n for n in nx.neighbors(graph,node)]

        numberOfNeigh=len(neighbours)

        trianglesCount=0

        # For every nodes, take 2 neighbour and find if there exist and edge between them
        if numberOfNeigh>1:
            for node1 in neighbours:
                for node2 in neighbours:
                    if graph.has_edge(node1,node2):
                        trianglesCount+=1

            clusteringCoefficent=trianglesCount/(numberOfNeigh*(numberOfNeigh-1))
            localClusteringDictionary[node] = clusteringCoefficent
    # End implementation of the algorithm

    endTime = time.time()
    
    print("LCC for each node: " + str(localClusteringDictionary))
    print("Computation time: " + str(endTime-startTime))
    
    LCCDataFrame = pd.DataFrame(localClusteringDictionary.items())
        
    #Plotting the results
    plt.bar(LCCDataFrame[0], LCCDataFrame[1])
    plt.xlabel('Airports')
    plt.ylabel('LCC')
    plt.show()

    

# Approximate Local Clustering Coefficent
def ApproximateLocalClusteringCoefficent(graph,k):
    # Counting the amount of time needed
    startTime = time.time()

    nodeList = nx.nodes(graph)
    numberOfNodes = len(nodeList)

    edgeList = nx.edges(graph)

    localClusteringDictionary = dict.fromkeys(nodeList, 0)
    nodesDictionary = dict.fromkeys(nodeList, 0)
    edgesDictionary = dict.fromkeys(edgeList, 0)
    
    # Start Implementation of the approximate algorithm for LCC
    for i in range(k):
        #Compute a permutation
        nodesPermutation = list(nodeList)
        random.shuffle(nodesPermutation)

        # Check the minimum
        for v in nodesPermutation:
            permutationIndex = numberOfNodes

            for neighbour in graph.neighbors(v):
                temporaryMin = nodesPermutation.index(neighbour)

                if(temporaryMin < permutationIndex):
                    permutationIndex = temporaryMin

            nodesDictionary[v] = permutationIndex
            
        for e in edgeList:
            if(nodesDictionary[e[0]] == nodesDictionary[e[1]]):
                edgesDictionary[e] = edgesDictionary[e] + 1

    #Compute the number of triangles
    for v in localClusteringDictionary.keys():
        sum = 0
        nodeDegree = graph.degree(v)

        for neighbour in graph.neighbors(v):
            uDegree = graph.degree(neighbour)

            if((v,neighbour) in edgesDictionary.keys()):
                sum = sum + (edgesDictionary[(v,neighbour)]/(edgesDictionary[(v,neighbour)]+k))*(uDegree+nodeDegree)
            else:
                sum = sum + (edgesDictionary[(neighbour,v)]/(edgesDictionary[(neighbour,v)]+k))*(uDegree+nodeDegree)

        if(nodeDegree != 1):
            if(sum*(1/(nodeDegree*(nodeDegree-1)))<1):
                localClusteringDictionary[v] = sum*(1/(nodeDegree*(nodeDegree-1)))
            else:
                localClusteringDictionary[v] = 1
    #End implementation of the algorithm
            
    endTime = time.time()
    
    print("Approximate LCC for each node: " + str(localClusteringDictionary))
    print("Computation time: " + str(endTime-startTime))
    
    approximateLCCDataFrame = pd.DataFrame(localClusteringDictionary.items())
        
    #Plotting the results
    plt.bar(approximateLCCDataFrame[0], approximateLCCDataFrame[1])
    plt.xlabel('Airports')
    plt.ylabel('Approximate LCC')
    plt.show()



# Computing the sub graph with the top N nodes for every kind of centrality measure
def SubGraphWithTopNodes(graph, centrality, n):
    # Sort in ascending order
    sortedTuples = sorted(centrality.items(), key=lambda item: item[1])

    centralitySorted = {k: v for k, v in sortedTuples}

    # Select the first N nodes
    firstNNodes = list(centralitySorted.items())[len(centralitySorted)-n:]

    newGraph = nx.Graph()

    # For all the couples of the top N nodes, we find the shortest path between them to construct the new graph
    for k1,v1 in firstNNodes:
        for k2,v2 in firstNNodes:
            #Finding a shortest path
            nodes_in_path = nx.shortest_path(graph, k1, k2)

            for i in range(len(nodes_in_path)-1):
                newGraph.add_node(nodes_in_path[i])

                if(i!=len(nodes_in_path)):
                    newGraph.add_edge(nodes_in_path[i],nodes_in_path[i+1])

    nx.draw(newGraph, with_labels = True)
    plt.show()
    #plt.savefig("topSubGraph.png", format="PNG")



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
