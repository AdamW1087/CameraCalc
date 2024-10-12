
def center_path(path):
    # if length of path is 0 it cannot be centered
    if len(path) == 0:
        return path
    
    # calculate the average x and average y
    centroid_x = sum(point[0] for point in path) / len(path)
    centroid_y = sum(point[1] for point in path) / len(path)
    
    # transform the points around the centre
    centered_path = [[point[0] - centroid_x, point[1] - centroid_y] for point in path]
    return centered_path

def min_max_scale(path):
    # get x and y coordinates from path
    x_coords = [point[0] for point in path]
    y_coords = [point[1] for point in path]
    
    # get min and max for all x's and all y's
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    # scale the path so all points are within 0-1
    scaled_path = [
        [(point[0] - x_min) / (x_max - x_min), (point[1] - y_min) / (y_max - y_min)]
        for point in path
    ]
    
    return scaled_path

# centre and scales the path to normalise it
def normalise_path(path):
    centered_path = center_path(path)
    normalised_path = min_max_scale(centered_path)
    return normalised_path

# processes a single path used for guessing
def process_path(path):
    normalised_path = normalise_path(path)
    pairPath = []
    for path in normalised_path:
        pairPath.append((path[0], path[1]))
        
    return pairPath