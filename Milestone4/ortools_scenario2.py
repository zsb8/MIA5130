from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2
distance_matrix = [
  [0, 5, 12, 13, 19, 28, 31, 31, 17, 15, 9, 9, 8, 9, 12, 19],
  [5, 0, 6, 6, 16, 25, 28, 28, 13, 11, 5, 5, 5, 5, 23, 21],
  [12, 6, 0, 1, 8, 15, 24, 30, 16, 14, 10, 11, 9, 8, 24, 29],
  [12, 6, 1, 0, 9, 16, 24, 31, 12, 11, 10, 12, 10, 6, 25, 30],
  [19, 15, 9, 10, 0, 12, 18, 20, 6, 7, 15, 15, 11, 11, 35, 33],
  [28, 24, 15, 16, 12, 0, 14, 21, 15, 17, 24, 24, 21, 21, 45, 43],
  [31, 27, 24, 29, 18, 14, 0, 9, 18, 20, 27, 27, 23, 23, 47, 45],
  [32, 28, 31, 30, 21, 24, 9, 0, 19, 21, 28, 28, 24, 25, 48, 47],
  [17, 13, 12, 12, 6, 14, 16, 16, 0, 3, 13, 13, 9, 9, 33, 32],
  [15, 11, 15, 11, 7, 17, 20, 20, 3, 0, 8, 8, 8, 8, 32, 30],
  [9, 5, 12, 12, 15, 24, 27, 27, 12, 10, 0, 1, 3, 4, 26, 24],
  [10, 5, 11, 10, 13, 22, 25, 25, 10, 6, 2, 0, 2, 3, 26, 24],
  [9, 4, 7, 7, 11, 21, 23, 23, 9, 7, 3, 2, 0, 1, 25, 23],
  [9, 5, 6, 6, 12, 21, 24, 24, 9, 7, 4, 3, 1, 0, 25, 23],
  [13, 17, 25, 25, 31, 41, 43, 43, 29, 27, 21, 21, 21, 21, 0, 3],
  [14, 22, 32, 32, 35, 44, 47, 47, 33, 31, 25, 25, 24, 25, 3, 0],
]

distance_matrix[6][7] = 999999999
distance_matrix[7][6] = 999999999

def create_data_model():
    return {
        "distance_matrix": distance_matrix,
        "num_vehicles": 1,
        "depot": 0 
    }

data = create_data_model()
manager = pywrapcp.RoutingIndexManager(len(data["distance_matrix"]), data["num_vehicles"], data["depot"])
routing = pywrapcp.RoutingModel(manager)
def distance_callback(from_index, to_index):
    from_node = manager.IndexToNode(from_index)
    to_node = manager.IndexToNode(to_index)
    return data["distance_matrix"][from_node][to_node]

transit_callback_index = routing.RegisterTransitCallback(distance_callback)
routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

search_parameters = pywrapcp.DefaultRoutingSearchParameters()
search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
solution = routing.SolveWithParameters(search_parameters)
if solution:
    index = routing.Start(0)
    route = []
    total_distance = 0
    while not routing.IsEnd(index):
        node = manager.IndexToNode(index)
        route.append(node)
        prev_index = index
        index = solution.Value(routing.NextVar(index))
        total_distance += routing.GetArcCostForVehicle(prev_index, index, 0)
    route.append(0) 
    print("Route order:", route)
    print("Total distance:", total_distance)
else:
    print("Can’t find solution.")


