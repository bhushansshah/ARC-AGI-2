import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from typing import List

def plot_task(
    task: List[dict],
    title: str = None
) -> None:
    """
    displays a task with grid dimensions labeled on axes
    """
    cmap = ListedColormap([
        '#000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
        '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'
    ])
    norm = Normalize(vmin=0, vmax=9)
    args = {'cmap': cmap, 'norm': norm}
    
    height = 2  # Always 2 rows (input and output)
    width = len(task)
    figure_size = (width * 3, height * 3)
    figure, axes = plt.subplots(height, width, figsize=figure_size)
    
    # Handle single example case
    if width == 1:
        axes = axes.reshape(2, 1)
    
    for column, example in enumerate(task):
        # Get grid dimensions
        input_height, input_width = len(example['input']), len(example['input'][0])
        output_height, output_width = len(example['output']), len(example['output'][0])
        
        # Plot the grids
        axes[0, column].imshow(example['input'], **args)
        axes[1, column].imshow(example['output'], **args)
        
        # Add dimension labels
        axes[0, column].set_title(f'Input: {input_height}×{input_width}', fontsize=10)
        axes[1, column].set_title(f'Output: {output_height}×{output_width}', fontsize=10)
        
        axes[0, column].axis('off')
        axes[1, column].axis('off')
    
    if title is not None:
        figure.suptitle(title, fontsize=20)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()