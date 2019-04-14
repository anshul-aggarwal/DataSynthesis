import random
import time
from operator import add
import matplotlib.pyplot as plt

start = time.time()

def distance(input1, input2):
    xor = bin(input1 ^ input2)
    dist = xor[2:].count('1')
    return dist

def run_kmeans(data, cluster_centres):
    clusters = [[] for _ in range(len(cluster_centres))]

    for i1 in data:
        if i1 in cluster_centres:
            clusters[cluster_centres.index(i1)].append(i1)
            continue
        distances = [distance(i1, c) for c in cluster_centres]
        closest = distances.index(min(distances))
        clusters[closest].append(i1)
    
    return clusters


def getnewcentres(clusters):
    centres = []
    for cluster in clusters:
        centre = [0 for _ in range(600)]
        for item in cluster:
            bin_rep = [int(x) for x in bin(item)[2:]]
            p = 599
            for l in range(len(bin_rep) - 1, -1, -1):
                centre[p] += bin_rep[l]
                p = p - 1
        cluster_size = len(cluster)
        if cluster_size == 0:
            centre = 0
        else:
            centre = int("".join([str(round(x/cluster_size)) for x in centre]), base=2)
        centres.append(centre)

    return centres


def k_means(data, cluster_count):
    initial_centres = random.sample(data, cluster_count)

    clusters = run_kmeans(data, initial_centres)
    print("Epoch completed:", 0, "Time:" ,round(time.time() - start, 2))

    epochs = 10
    for e_ in range(epochs):
        new_centres = getnewcentres(clusters)
        if set(new_centres) == set(initial_centres):
            break
        clusters = run_kmeans(data, new_centres)
        initial_centres = new_centres
        print("Epoch completed:", e_ + 1, "Time:" ,round(time.time() - start, 2))

    return clusters, initial_centres


def max_distance(clusters, cluster_centres):
    max_dists = [0 for _ in range(100)]
    tq_dists = [-1 for _ in range(100)]

    for i in range(len(clusters)):
        max_dist = -1
        dists = []
        for item in clusters[i]:
            dist = distance(item, cluster_centres[i])
            dists.append(dist)
            if dist > max_dist:
                max_dist = dist
        max_dists[i] = max_dist
        try:
            dists.sort()
            tq_dists[i] = dists[int(len(dists)*3/4)]
        except:
            continue
    return max_dists, tq_dists


def find_intra_cluster_max_distance(cluster):
    max_dist = -1
    for x in cluster:
        for y in cluster:
            dist = distance(x,y)
            if dist > max_dist:
                max_dist = dist

    return max_dist


def find_dist_distribution(cluster, clus_centre, counter):
    distances = [distance(y, clus_centre) for y in cluster]
    plt.figure(counter)
    plt.hist(distances, bins=30)
    plt.savefig(str(counter) + ".png")


def calculate_summary_statistics(data):
    stats = [0 for _ in range(600)]
    for item in data:
        bin_rep = [int(x) for x in bin(item)[2:]]
        p = 599
        for l in range(len(bin_rep) - 1, -1, -1):
            stats[p] += bin_rep[l]
            p -= 1
    datasize = len(data)
    stats = [x/datasize for x in stats]
    return stats




def initialize():
    f = open("dataset").read().strip().split()[:10000]

    #Converting to integer for better binary handling
    for i in range(len(f)):
        f[i] = int(f[i].replace(",",""), base=2)

    print(calculate_summary_statistics(f)[1:20])

    clusters, clus_centres = k_means(f, 100)
    #clus_lengths = [len(x) for x in clusters]
    #print(clus_lengths)
    #print(sum(clus_lengths))
    max_dists, tq_dists = max_distance(clusters, clus_centres)

    #print(tq_dists)

    # x = random.sample(range(100), 5)
    # for _x in x:
    #     find_dist_distribution(clusters[_x], clus_centres[_x], x.index(_x))
    


if __name__ == "__main__":
    initialize()