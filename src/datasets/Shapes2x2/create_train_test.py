import torch
import numpy as np
import math

pic_size=112

# General
size_dict = {
    's': 20,
    'm': 40,
    'l': 60
}

color_dict = {
    'r': 'red',
    'g': 'green',
    'b': 'blue'
}

shape_dict = {
    'c': 'circle',
    'q': 'square',
    't': 'triangle'
}

all_letter_combs = [i+j+k for i in color_dict.keys() for j in shape_dict.keys() for k in size_dict.keys()]
all_label_combs = [i+j+k+l for i in all_letter_combs for j in all_letter_combs for k in all_letter_combs for l in all_letter_combs]
    

# Creating shapes
# Circles

def create_colored_circle(color='red', size=100, side_length=30, center=None):
    """
    Creates a colored circle on a black background using PyTorch.
    
    Args:
        color (str): Color of the circle ('red', 'green', or 'blue')
        size (int): Size of the square image in pixels
        radius (int): Radius of the circle in pixels
        center (tuple): Optional center coordinates, defaults to image center
    
    Returns:
        torch.Tensor: RGB tensor of shape (3, size, size)
    """

    radius = side_length // 2
    if center is None:
        center = (size // 2, size // 2)
    
    # Validate color input
    color = color.lower()
    if color not in ['red', 'green', 'blue']:
        raise ValueError("Color must be 'red', 'green', or 'blue'")
    
    # Create coordinate grids
    y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
    
    # Calculate distances from center
    distances = torch.sqrt((x - center[0])**2 + (y - center[1])**2)
    
    # Create the circle mask
    circle_mask = distances <= radius
    
    # Create RGB tensor (3, height, width)
    image = torch.zeros(3, size, size)
    
    # Set the appropriate color channel to 1 where the circle mask is True
    color_channel = {'red': 0, 'green': 1, 'blue': 2}
    image[color_channel[color]][circle_mask] = 1.0
    
    return image

# Squares
def create_colored_square(color='red', size=100, side_length=30, center=None):
    """
    Creates a colored square on a black background using PyTorch.
    
    Args:
        color (str): Color of the square ('red', 'green', or 'blue')
        size (int): Size of the square image in pixels
        side_length (int): Length of the square's side in pixels
        center (tuple): Optional center coordinates, defaults to image center
    
    Returns:
        torch.Tensor: RGB tensor of shape (3, size, size)
    """
    if center is None:
        center = (size // 2, size // 2)
    
    # Validate color input
    color = color.lower()
    if color not in ['red', 'green', 'blue']:
        raise ValueError("Color must be 'red', 'green', or 'blue'")
    
    # Calculate square boundaries
    half_side = side_length // 2
    x_min = center[0] - half_side
    x_max = center[0] + half_side
    y_min = center[1] - half_side
    y_max = center[1] + half_side
    
    # Create coordinate grids
    y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
    
    # Create the square mask
    square_mask = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
    
    # Create RGB tensor (3, height, width)
    image = torch.zeros(3, size, size)
    
    # Set the appropriate color channel to 1 where the square mask is True
    color_channel = {'red': 0, 'green': 1, 'blue': 2}
    image[color_channel[color]][square_mask] = 1.0
    
    return image

# Triangle
def create_colored_triangle(color='red', size=100, side_length=30, center=None):
    """
    Creates a colored equilateral triangle on a black background using PyTorch.
    
    Args:
        color (str): Color of the triangle ('red', 'green', or 'blue')
        size (int): Size of the square image in pixels
        side_length (int): Length of the triangle's side in pixels
        center (tuple): Optional center coordinates, defaults to image center
    
    Returns:
        torch.Tensor: RGB tensor of shape (3, size, size)
    """
    if center is None:
        center = (size // 2, size // 2)
    
    # Validate color input
    color = color.lower()
    if color not in ['red', 'green', 'blue']:
        raise ValueError("Color must be 'red', 'green', or 'blue'")
    
    # Calculate triangle vertices
    # For an equilateral triangle, height = side_length * sqrt(3)/2
    height = side_length * math.sqrt(3) / 2
    half_side = side_length / 2
    
    # Define the three vertices (x, y)
    # Center point is the centroid of the triangle
    top = (center[0], center[1] - (2/3) * height)
    bottom_left = (center[0] - half_side, center[1] + (1/3) * height)
    bottom_right = (center[0] + half_side, center[1] + (1/3) * height)
    
    # Create coordinate grids
    y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
    x = x.float()
    y = y.float()
    
    # Function to determine if a point is inside the triangle using barycentric coordinates
    def point_in_triangle(px, py, v1, v2, v3):
        def sign(p1x, p1y, p2x, p2y, p3x, p3y):
            return (p1x - p3x) * (p2y - p3y) - (p2x - p3x) * (p1y - p3y)
        
        d1 = sign(px, py, v1[0], v1[1], v2[0], v2[1])
        d2 = sign(px, py, v2[0], v2[1], v3[0], v3[1])
        d3 = sign(px, py, v3[0], v3[1], v1[0], v1[1])
        
        has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
        has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
        
        return ~(has_neg & has_pos)
    
    # Create the triangle mask
    triangle_mask = point_in_triangle(x, y, top, bottom_left, bottom_right)
    
    # Create RGB tensor (3, height, width)
    image = torch.zeros(3, size, size)
    
    # Set the appropriate color channel to 1 where the triangle mask is True
    color_channel = {'red': 0, 'green': 1, 'blue': 2}
    image[color_channel[color]][triangle_mask] = 1.0
    
    return image

# Shape (with randomised center)
def create_shape(color, shape, size):
    color = color_dict[color]
    size = size_dict[size]
    shape = shape_dict[shape]
    center = (pic_size//2 + np.random.randint(-18,19), pic_size//2 + np.random.randint(-18,19))
    if shape == 'circle':
        return create_colored_circle(color=color, size=pic_size, side_length=size, center=center)
    elif shape == 'square':
        return create_colored_square(color=color, size=pic_size, side_length=size, center=center)
    elif shape == 'triangle':
        return create_colored_triangle(color=color, size=pic_size, side_length=size, center=center)
    else:
        raise ValueError("Shape must be 'circle', 'square', or 'triangle'")
    return shape

# Create dataset utils
def merge2x2(t0,t1,t2,t3):
    return torch.cat((torch.cat((t0,t1),dim=1),torch.cat((t2,t3),dim=1)),dim=0)

# Create shapes dataset
def create_gt_labels(size=1000):
    labels = np.empty(size, dtype='U12')
    colors = ['r', 'g', 'b']
    shapes = ['c', 'q', 't']
    sizes = ['s', 'm', 'l']
    random_colors=np.random.choice(colors, size=size*4)
    random_shapes=np.random.choice(shapes, size=size*4)
    random_sizes=np.random.choice(sizes, size=size*4)
    bigstring=''.join(i+j+k for i,j,k in zip(random_colors,random_shapes,random_sizes))
    labels = np.array([bigstring[i:i + 12] for i in range(0, len(bigstring), 12)])
    return labels

def create_tensors_from_labels(labels):
    data = torch.empty(len(labels), 3, 2*pic_size, 2*pic_size)
    for i in range(len(labels)):
        str = labels[i]
        for j in range(4):
            color = str[j*3]
            shape = str[j*3+1]
            size = str[j*3+2]
            data[i, :, (j//2)*pic_size:((j//2)+1)*pic_size, (j%2)*pic_size:((j%2)+1)*pic_size] = create_shape(color, shape, size)
    return data


