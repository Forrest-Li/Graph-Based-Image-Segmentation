import random
import numpy as np
from sklearn.neighbors import KDTree, NearestNeighbors

from graph_segment import graph_segment


def random_color(seed):
    random.seed(seed+10)
    return random.choices(range(256), k=3)


def image_segment(img, k, min_size):
    """
    Performing image segmentations
    :param img: input pre-processed image
    :param k: threshold function parameter
    :param min_size: min. size to be recognized as a component
    :return: mask of image, diff. components represented by diff. colors
    """

    img_height = img.shape[0]
    img_width = img.shape[1]
    # Build graph
    edges = []  # dict of edges, `edges`: list of `edge`s, `edge` format: {'w': int, 'a': int, 'b': int}
    iter_gen = ((y, x) for y in range(img_height) for x in range(img_width))

    # Method 2: Finding k nearest neighbor (KNN, KDTree)
    # Better | params(beach): sigma: 0.9; k: 2000; min_size: 100; number of neighbors: 10
    # Best   | params(beach): HSV color; sigma: 0.9; k: 2000; min_size: 100; number of neighbors: 10
    # Fair   | params(grain): HSV color; sigma: 0.9; k: 3000; min_size: 100; number of neighbors: 15
    # """
    num_nbr = 15
    img_flat = img.reshape(img_width * img_height, 3)
    # knn = NearestNeighbors(n_neighbors=num_nbr).fit(img_flat)  # KNN: slower, similar performance as KDTree
    tree = KDTree(img_flat)  # KDTree: faster

    for (g_y, g_x) in iter_gen:
        # near_nbrs = knn.kneighbors([img[g_y, g_x]], return_distance=False)  # KNN
        near_nbrs = tree.query([img[g_y, g_x]], k=num_nbr, return_distance=False)  # KDTree
        for nbr in near_nbrs[0]:
            nbr_height = nbr // img_width
            nbr_width = nbr % img_width
            edges.append({'a': g_y * img_width + g_x,
                          'b': nbr,
                          'w': np.linalg.norm(img[g_y, g_x] - img[nbr_height, nbr_width])})
    # """

    # Method 1: Finding 8-direction nearest neighbor
    # Better | params(beach): sigma: 0.9; k: 700; min_size: 100
    # Best   | params(beach): HSV color; sigma: 0.9; k: 1000; min_size: 100
    # Good   | params(grain): sigma: 0.9; k: 1500; min_size: 100
    """
    for (g_y, g_x) in iter_gen:
        if g_x < img_width - 1:  # value to the right exists
            edges.append({'a': g_y * img_width + g_x,
                          'b': g_y * img_width + g_x + 1,
                          'w': np.linalg.norm(img[g_y, g_x] - img[g_y, g_x + 1])})

        if g_y < img_height - 1:  # value to the bottom exists
            edges.append({'a': g_y * img_width + g_x,
                          'b': (g_y + 1) * img_width + g_x,
                          'w': np.linalg.norm(img[g_y, g_x] - img[g_y + 1, g_x])})

        if (g_x < img_width - 1) and (g_y < img_height - 1):  # value to the bottom-right exists
            edges.append({'a': g_y * img_width + g_x,
                          'b': (g_y + 1) * img_width + g_x + 1,
                          'w': np.linalg.norm(img[g_y, g_x] - img[g_y + 1, g_x + 1])})

        if (g_x < img_width - 1) and (g_y >= 1):  # value to the top-right exists
            edges.append({'a': g_y * img_width + g_x,
                          'b': (g_y - 1) * img_width + g_x + 1,
                          'w': np.linalg.norm(img[g_y, g_x] - img[g_y - 1, g_x + 1])})
    """

    # Perform graph segmentations
    num_pxls = img_width * img_height
    num_edges = len(edges)
    u = graph_segment(num_pxls, num_edges, edges, k)

    # Small components post-processing
    for i in range(num_edges):  # join small components
        a = u.trace(edges[i]['a'])
        b = u.trace(edges[i]['b'])
        if (a != b) and ((u.get_size_of(a) < min_size) or (u.get_size_of(b) < min_size)):
            u.join(a, b)

    del edges

    img_mask = np.zeros((img_height, img_width, 3))
    iter_gen = ((y, x) for y in range(img_height) for x in range(img_width))
    for (g_y, g_x) in iter_gen:  # color each component with diff. colors
        comp = u.trace(g_y * img_width + g_x)
        img_mask[g_y, g_x] = random_color(comp)
    img_mask = img_mask.astype(np.uint)

    del u

    return img_mask
