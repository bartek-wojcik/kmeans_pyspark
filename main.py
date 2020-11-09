from pyspark import SparkConf, SparkContext


def map_line_to_vector(line):
    return line.split()


conf = SparkConf()
sc = SparkContext(conf=conf)
lines = sc.textFile('3a.txt')
vectors = lines.map(map_line_to_vector)
