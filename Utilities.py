import numpy as np

def count_unique_value(col):
    """Count unique values and coressponding numbers of each value.

    Args:
        col (DataFrame): for each column in dataset.
    """
    col = col.values.tolist()
    visited = {}
    for i in range(1, len(col)):  # without header
        if str(col[i]) not in visited:
            visited.update({str(col[i]): 1})
        else:
            visited[str(col[i])] += 1
    
    return visited
                