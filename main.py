from pyspark import SparkConf, SparkContext
from math import sqrt

MAX_ITERATIONS = 20
CLUSTERS = 10


def map_line_to_vector(line):
    return tuple(map(float, line.split()))


def map_line_to_centroid(line):
    return map_line_to_vector(line), ()


conf = SparkConf()
sc = SparkContext(conf=conf)

# Read vectors
lines = sc.textFile('3a.txt')
vectors = lines.map(map_line_to_vector)

# Read random centroids
lines = sc.textFile('3b.txt')
random_centroids = lines.map(map_line_to_vector).collect()

# Read farthest centroids
lines = sc.textFile('3c.txt')
farthest_centroids = lines.map(map_line_to_vector).collect()


def assign_point_to_centroid(point, centroids, distance_function):
    distances = list(map(lambda centroid: distance_function(centroid, point), centroids))
    return distances.index(min(distances)), point


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


def kmeans(vectors, centroids, distance_function):
    for iteration in range(MAX_ITERATIONS):
        assignments = vectors.map(lambda vector: assign_point_to_centroid(vector, centroids, distance_function))
        centroids = assignments.groupByKey().map(lambda group: map_group_to_centroid(group)).collect()
    print_centroids(centroids)


kmeans(vectors, random_centroids, euclidean_distance)
