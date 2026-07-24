import numpy as np
import random
import time
import matplotlib.pyplot as plt

# Route cost
def tsp_route_cost(route, dist):
   return np.sum(dist[route[:-1], route[1:]]) + dist[route[-1], route[0]]

# Check validity
def route_test(route, dist):
    n = dist.shape[0]
    return np.array_equal(np.sort(route), np.arange(n))

# Simulated Annealing TSP
def solve_tsp(incidence, TIME, temperature, cooling):

    n = incidence.shape[0]

    route = np.random.permutation(n)
    current_cost = tsp_route_cost(route, incidence)

    best_route = route.copy()
    best_cost = current_cost

    history = []
    iterations = 0

    start_time = time.time()

    while time.time() - start_time < TIME:

        for _ in range(1000):

            move = random.choices(
                ["relocate", "swap", "2-opt"],
                weights=[0.33, 0.33, 0.34]
            )[0]

            if move == "relocate":

                idx1 = random.randrange(n)
                node = route[idx1]

                newroute = np.delete(route, idx1)

                idx2 = random.randrange(n)

                newroute = np.insert(newroute, idx2, node)

                iterations += 1

            elif move == "swap":

                idx1, idx2 = random.sample(range(n), 2)

                newroute = route.copy()

                newroute[idx1], newroute[idx2] = (
                    newroute[idx2],
                    newroute[idx1],
                )

            else:      # 2-opt

                idx1, idx2 = sorted(random.sample(range(n), 2))

                newroute = route.copy()

                newroute[idx1:idx2 + 1] = newroute[idx1:idx2 + 1][::-1]

            new_cost = tsp_route_cost(newroute, incidence)

            cost_diff = new_cost - current_cost

            alpha = random.random()

            if cost_diff <= 0:

                route = newroute
                current_cost = new_cost

                if new_cost < best_cost:
                    best_cost = new_cost
                    best_route = newroute.copy()

            elif np.exp(-cost_diff / temperature) > alpha:

                route = newroute
                current_cost = new_cost

                if new_cost < best_cost:
                    best_cost = new_cost
                    best_route = newroute.copy()

            temperature *= cooling

            if iterations % 100 == 0:
                history.append(best_cost)

    return {
        "route": best_route,
        "history": history,
        "best_cost": best_cost,
        "iterations": iterations,
    }


def nearest_neighbour_cvrp(coordinates, demands, vechile_capacity):
    
    unvisited = set(coordinates.keys())
    unvisited.remove(0)
    
    routes = []
    
    while unvisited:
        
        route = [0]
        load = 0
        current = 0
        
        while True:
            
            feasible = [
                node for node in unvisited
                if load + demands[node] <= vehicle_capacity
            ]
            
            if not feasible:
                break
            
            next_node = min(
                feasible, 
                key=lambda node:distance_matrix[(current,node)]
            )
            
            route.append(next_node)
            load += demands[next_node]
            current = next_node
            unvisited.remove(next_node)
            
        route.append(0)
        routes.append((route,load))
        
    return routes

def distance(i, j):
    x1, y1 = coordinates[i]
    x2, y2 = coordinates[j]
    return math.hypot(x2 - x1, y2 - y1)

# set depot as first and last vertex in route
def init_depot(route):
    depot_pos = route.index(0)
    route = route[depot_pos:] + route[:depot_pos]
    route.append(0)
    return route

def sa_split_CVRP():
    n = len(coordinates)

    # all customers no depot
    nodes = list(range(1, n))

    sub_dist = distance_matrix[np.ix_(nodes, nodes)]

    # route first via sa
    sol = solve_tsp(sub_dist, TIME=0.5*n, temperature=250, cooling=0.999)

    big_route = [nodes[node] for node in sol["route"]]

    routes = []
    current_route = [0]      # start at depot
    current_load = 0

    # cluster by splitting 
    for node in big_route:

        if current_load + demands[node] <= vehicle_capacity:
            current_route.append(node)
            current_load += demands[node]

        else:
            # return to depot
            current_route.append(0)
            routes.append((current_route, current_load))

            # new route
            current_route = [0, node]
            current_load = demands[node]

    current_route.append(0)
    routes.append((current_route, current_load))

    return routes

def sweep_sa_CVRP():
    # cluster first
    clusters = sweep_clusters(
    coordinates,
    demands,
    vehicle_capacity
)

    for cluster in clusters:
        assert sum(demands[i] for i in cluster) <= vehicle_capacity

    routes = []

    for cluster in clusters:

        nodes = [0] + cluster

        sub_dist = distance_matrix[np.ix_(nodes,nodes)]

        # SA parameters based on cluster size
        cluster_size = len(nodes)

        temperature = 20 * cluster_size

        if cluster_size <= 10:
            cooling = 0.995
        elif cluster_size <= 30:
            cooling = 0.997
        else:
            cooling = 0.999

        sol = solve_tsp(sub_dist, TIME=0.5*cluster_size, temperature=temperature, cooling=cooling)

        route = [nodes[i] for i in sol["route"]]
        route = init_depot(route)

        load = sum(demands[node] for node in cluster)

        routes.append((route, load))

    return routes

def sweep_clusters(coordinates, demands, vehicle_capacity):
    depot = coordinates[0]
    clusters = []
    nodes = []

    # Calculate angle of each node relative to depot
    for node in range(1, len(coordinates)):

        x, y = coordinates[node]

        angle = math.atan2(
            y - depot[1],
            x - depot[0]
        )

        nodes.append((node, angle))


    # Sort nodes by angle around depot
    nodes.sort(key=lambda x: x[1])

    # start at random node
    start = random.randint(0, len(nodes)-1)
    nodes = nodes[start:] + nodes[:start]

    cluster = []
    load = 0

    # Sweep around depot
    for node, angle in nodes:

        demand = demands[node]

        # If customer does not fit, get new vehicle
        if load + demand > vehicle_capacity:

            clusters.append(cluster)

            cluster = []
            load = 0


        cluster.append(node)
        load += demand

    # Add final cluster
    if cluster:
        clusters.append(cluster)

    return clusters

def nn_sa_CVRP():
    # cluster first
    clusters = nearest_neighbour_clusters(
    coordinates,
    demands,
    vehicle_capacity
)

    for cluster in clusters:
        assert sum(demands[i] for i in cluster) <= vehicle_capacity

    routes = []

    for cluster in clusters:

        nodes = [0] + cluster

        sub_dist = distance_matrix[np.ix_(nodes,nodes)]

        # SA parameters based on cluster size
        cluster_size = len(nodes)

        temperature = 20 * cluster_size

        if cluster_size <= 10:
            cooling = 0.995
        elif cluster_size <= 30:
            cooling = 0.997
        else:
            cooling = 0.999

        sol = solve_tsp(sub_dist, TIME=0.5*cluster_size, temperature=temperature, cooling=cooling)

        route = [nodes[i] for i in sol["route"]]
        route = init_depot(route)

        load = sum(demands[node] for node in cluster)

        routes.append((route, load))

    return routes

def nearest_neighbour_clusters(coordinates, demands, vehicle_capacity):

    nodes = set(range(1, len(coordinates)))  # no depot
    clusters = []

    while nodes:

        cluster = []
        load = 0
        current_node = 0  # start from depot


        while True:

            feasible = [
                node for node in nodes
                if load + demands[node] <= vehicle_capacity
            ]

            if not feasible:
                break


            # choose nearest feasible node
            next_node = min(
                feasible,
                key=lambda node: distance(current_node, node)
            )


            cluster.append(next_node)

            load += demands[next_node]

            nodes.remove(next_node)

            current_node = next_node


        clusters.append(cluster)


    return clusters