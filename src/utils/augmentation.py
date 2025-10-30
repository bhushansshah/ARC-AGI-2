import numpy as np

# --------------------------
# 1. Rotation
# --------------------------
def rotate_grid(grid, angle):
    """
    Rotate a grid (numpy array) by 90, 180, or 270 degrees.
    Args:
        grid (np.ndarray): Input grid.
        angle (int): Rotation angle in degrees. Must be one of [90, 180, 270].
    Returns:
        np.ndarray: Rotated grid.
    """
    if angle not in [90, 180, 270]:
        raise ValueError("Angle must be 90, 180, or 270")
    k = angle // 90
    return np.rot90(grid, k=k)  # rot90 rotates counter-clockwise

# --------------------------
# 2. Color Change
# --------------------------
def change_color(grid, color_map=None):
    """
    Change colors/symbols in the grid consistently.
    Args:
        grid (np.ndarray): Input grid.
        color_map (dict): Optional dict mapping old_color -> new_color.
                          If None, random mapping is created.
    Returns:
        np.ndarray: Grid with colors changed.
    """
    grid = np.array(grid)
    unique_colors = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    if color_map is None:
        shuffled_colors = np.random.permutation(unique_colors)
        color_map = {old: new for old, new in zip(unique_colors, shuffled_colors)}
    
    new_grid = grid.copy()
    for old_color, new_color in color_map.items():
        new_grid[grid == old_color] = new_color
    return new_grid, color_map

# --------------------------
# 3. Horizontal Flip
# --------------------------
def flip_horizontal(grid):
    """
    Flip grid horizontally.
    Args:
        grid (np.ndarray): Input grid.
    Returns:
        np.ndarray: Horizontally flipped grid.
    """
    return np.fliplr(grid)

# --------------------------
# 4. Vertical Flip
# --------------------------
def flip_vertical(grid):
    """
    Flip grid vertically.
    Args:
        grid (np.ndarray): Input grid.
    Returns:
        np.ndarray: Vertically flipped grid.
    """
    return np.flipud(grid)

# --------------------------
# 5. Padding Augmentation
# --------------------------
def padding_grid(training_sample_input, padding_top, padding_bottom, padding_left, padding_right, padding_value=0):
    """
    Pad the grid with specified padding on each side.
    Args:
        training_sample_input (np.ndarray): Input grid.
        padding_top (int): Number of rows to pad on top.
        padding_bottom (int): Number of rows to pad at bottom.
        padding_left (int): Number of columns to pad on left.
        padding_right (int): Number of columns to pad on right.
        padding_value (int): Value to use for padding.
    Returns:
        np.ndarray: Padded grid.
    """
    padded_grid = np.pad(training_sample_input, 
                         ((padding_top, padding_bottom), (padding_left, padding_right)), 
                         mode='constant', constant_values=padding_value)
    return padded_grid

def horizontal_scale_grid(grid):
    """
    Scale the grid horizontally by a factor of 2.
    Args:
        grid (np.ndarray): Input grid.
    Returns:
        np.ndarray: Horizontally scaled grid.
    """
    return np.repeat(grid, 2, axis=1)

def vertical_scale_grid(grid):
    """
    Scale the grid vertically by a factor of 2.
    Args:
        grid (np.ndarray): Input grid.
    Returns:
        np.ndarray: Vertically scaled grid.
    """
    return np.repeat(grid, 2, axis=0)