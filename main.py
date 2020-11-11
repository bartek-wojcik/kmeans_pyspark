from pyspark import SparkConf, SparkContext
from math import sqrt
from matplotlib import pyplot as plt

MAX_ITERATIONS = 20
CLUSTERS = 10


def map_line_to_vector(line):
    return tuple(map(float, line.split()))


conf = SparkConf()
sc = SparkContext(conf=conf)

# Read vectors
lines = sc.textFile('3a.txt')
vectors = lines.map(map_line_to_vector)

# Read random centroids
lines = sc.textFile('3b.txt')
random_centroids = lines.map(map_line_to_vector).collect()[:CLUSTERS]

# Read farthest centroids
lines = sc.textFile('3c.txt')
farthest_centroids = lines.map(map_line_to_vector).collect()[:CLUSTERS]


def vector_difference(point, centroid):
    return [e1 - e2 for e1, e2 in zip(point, centroid)]


def assign_point_to_centroid(point, centroids, distance_function, cost_function):
    distances = list(map(lambda centroid: distance_function(centroid, point), centroids))
    min_index = distances.index(min(distances))
    cost = cost_function(centroids[min_index], point)
    return min_index, point, cost


def manhattan_distance(centroid, point):
    return sum(abs(e1 - e2) for e1, e2 in zip(centroid, point))


def euclidean_distance(centroid, point):
    return sqrt(sum((e1 - e2) ** 2 for e1, e2 in zip(centroid, point)))


def map_group_to_centroid(group_tuple):
    groups = list(group_tuple[1])
    new_centroid = groups[0]
    for group in groups[1:]:
        new_centroid = [sum(x) for x in zip(new_centroid, group)]
    return tuple([x / len(groups) for x in new_centroid])


def print_centroids(centroids):
    for centroid in centroids:
        print(centroid)


def cost_euclidean(centroid, point):
    vector = vector_difference(point, centroid)
    return sqrt(sum(e ** 2 for e in vector)) ** 2


def cost_manhattan(centroid, point):
    vector = vector_difference(point, centroid)
    return sqrt(sum(e ** 2 for e in vector)) ** 2


def kmeans(vectors, centroids, distance_function, cost_function):
    costs = []
    for iteration in range(MAX_ITERATIONS):
        assignments = vectors.map(
            lambda vector: assign_point_to_centroid(vector, centroids, distance_function, cost_function))
        cost = assignments.map(lambda assigment: assigment[2]).sum()
        costs.append(cost)
        centroids = assignments.map(lambda assigment: (assigment[0], assigment[1])).groupByKey().map(
            lambda group: map_group_to_centroid(group)).collect()
    # print_centroids(centroids)
    print(f'Difference between 1 and 10 iteration: {costs[0] / costs[9] * 100}%')
    iterations = range(1, MAX_ITERATIONS + 1)
    plt.plot(iterations, costs)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.xticks(iterations)
    plt.show()


kmeans(vectors, random_centroids, manhattan_distance, cost_manhattan)
kmeans(vectors, random_centroids, euclidean_distance, cost_euclidean)
kmeans(vectors, farthest_centroids, manhattan_distance, cost_manhattan)
kmeans(vectors, farthest_centroids, euclidean_distance, cost_euclidean)
