from universe import Universe

def graph_segment(num_pxls, num_edges, edges, k):
    """
    Perform graph segmentations
    :param num_pxls: number of pixels in original img
    :param num_edges: number of edges created
    :param edges: pair of edges
    :param k: tau function param.
    :return: created Universe instance
    """
    # Sort edges
    edges = sorted(edges, key=lambda e: (e['w']))

    # Initialize thresholds
    threshold = [k / 1] * num_pxls

    # Initialize disjoint-set forest
    u = Universe(num_pxls)

    # Iterate all edges (non-decreasing order)
    for edge in edges:
        a = u.trace(edge['a'])
        b = u.trace(edge['b'])

        if a != b:
            if (edge['w'] < threshold[a]) and (edge['w'] < threshold[b]):
                u.join(a, b)
                a = u.trace(a)
                threshold[a] = edge['w'] + k / u.get_size_of(a)

    del threshold

    return u
