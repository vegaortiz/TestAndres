import numpy as np
import pandas as pd

def read_excel_file(filename, sheet_name): #Read coordinates and demand values from a specific sheet in an Excel file. Assumes the data is in columns labeled 'X', 'Y', and 'Demand'.
    df = pd.read_excel(filename, sheet_name=sheet_name, header=1) #ponemos header=1 para poner la primera fila de como 
    print(df)
    coordinates = df[['X','Y']].values
    demands = df['Demanda'].values
    return coordinates, demands


def calculate_distance_matrix(coordinates): #Calculate the distance matrix between coordinates.
    num_points = len(coordinates)
    dist_matrix = np.zeros((num_points, num_points))

    for i in range(num_points):
        for j in range(num_points):
            dist_matrix[i, j] = calculate_distance(coordinates, i, j)
    return dist_matrix

def calculate_distance(coordinates, i, j): #Calculate the Euclidean distance between two points.
    x1, y1 = coordinates[i]
    x2, y2 = coordinates[j]
    return np.sqrt((x1 - x2) **2+ (y1 - y2) **2)


def calculate_total_distance(route, dist_matrix): #Calculate the total distance of a given route using the distance matrix.
    total_distance = 0
    num_points =len(route)
    for i in range(num_points -1):
        current_node = route[i]
        next_node = route[i +1]
        total_distance += dist_matrix[current_node, next_node]
    return total_distance

def nearest_neighbor(dist_matrix, demands, capacity):#Apply the Nearest Neighbor heuristic to find initial routes for VRP.
    num_points = dist_matrix.shape[0]
    visited = np.zeros(num_points, dtype=bool)
    routes = []

    while np.sum(visited) < num_points:
        current_node = 0 # Start at node 0
        current_capacity = 0
        route = [current_node]
        visited[current_node] = True 

        while current_capacity + demands[current_node] <= capacity:
            current = route[-1]
            nearest = None
            min_dist = float('inf')

            for neighbor in np.where(~visited)[0]:
                if demands[neighbor] + current_capacity <= capacity and dist_matrix:
                    nearest = neighbor
                    min_dist = dist_matrix[current, neighbor]
            if nearest is None:
                break


            route.append(nearest)
            visited[nearest] = True
            current_capacity += demands[nearest]
        routes.append(route)
    return routes



def format_output(routes):
    #Format the final routes as required. In this example, it returns a list of routes.
    return routes


def vrp_solver(filename, sheet_name, capacity):# Solve the VRP using the provided filename for coordinates and vehicle capac 
    coordinates, demands = read_excel_file(filename, sheet_name)
    dist_matrix = calculate_distance_matrix(coordinates)
    routes = nearest_neighbor(dist_matrix, demands, capacity) 
    formatted_routes = format_output(routes)
    return formatted_routes

#Use nearest neighbor
filename = r"D:\uni CEU\segundo cuatri\proyecto 1\excel coordendas\ubicaciones exactas península.xlsx" #Copy file pa 

print(filename)

sheet_name = "Hoja1" # Specify the name of the sheet or its index
capacity = 2010 # Specify the capacity of the vehicle
solution = vrp_solver(filename, sheet_name, capacity)


for route in solution: 
    print(route)



def two_opt(routes, dist_matrix, num_iterations):
    best_routes = routes.copy()

    for _ in range(num_iterations):
        selected_route_idx = np.random.randint(0,len(routes))
        selected_route = routes[selected_route_idx]

        i, j = np.random.randint(1,len(selected_route) -1, size=2)

        if j < i:
            i, j = j, i

        new_route = selected_route.copy()
        new_route[i:j] = selected_route[j -1: i - 1: -1] # Reverse the path b

        new_routes = routes.copy()
        new_routes[selected_route_idx] = new_route


        if calculate_total_distance(new_routes[selected_route_idx], dist_matrix, best_routes[selected_route_idx], dist_matrix):
            best_routes = new_routes

    return best_routes


def vrp_solver2(filename, sheet_name, capacity, num_iterations): #Solve the VRP using the provided filename for coordinates, vehicle capacity and number of iterations for the two-opt optimization.
    coordinates, demands = read_excel_file(filename, sheet_name)
    dist_matrix = calculate_distance_matrix(coordinates)
    routes = nearest_neighbor(dist_matrix, demands, capacity)

    for i in range(len(routes)):
        route = routes[i]
        optimized_route = two_opt([route], dist_matrix, num_iterations)[0]
        routes[i] = optimized_route

    formatted_routes = format_output(routes)
    return formatted_routes