"""
 ============================================================================
 Name        : create_gradient_on_image.py
 Author      : Sam Lunev
 Version     : .0
 Copyright   : All rights reserved
 Description : Block 1 CV-32 task
 ============================================================================
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
from enum import Enum


class Direction(Enum):
    """
    Enumeration for gradient directions.
    Attributes:
        FORWARD: Represents forward direction (0 to 1)
        REVERSE: Represents reverse direction (1 to 0)
    """

    FORWARD = "forward"
    REVERSE = "reverse"

def create_coordinate_grids(width: int = 256, height: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates two-dimensional coordinate grids for gradient generation.
    Args:
        width (int): Width of the grid in pixels. Default is 256.
        height (int): Height of the grid in pixels. Default is 256.
    Returns:
        tuple: Two arrays representing X and Y coordinates
    """

    if not isinstance(width, int):
        raise TypeError("Expected 'width' to be an integer")
    if not isinstance(height, int):
        raise TypeError("Expected 'height' to be an integer")

    try:
        x = np.linspace(0, 1, width)
        y = np.linspace(0, 1, height)
        X, Y = np.meshgrid(x, y)
        return X, Y
    except Exception as e:
        print(f"Exception: {e}")
        sys.exit(1)  # Terminate immediately

def create_black_to_white_gradient(width: int = 256, height: int = 256) -> np.ndarray:
    """
    Creates a black to white linear gradient.
    Args:
        width (int): Width of the gradient in pixels. Default is 256.
        height (int): Height of the gradient in pixels. Default is 256.
    Returns:
        np.ndarray: Gradient array with values from 0 (black) to 1 (white)
    """

    if not isinstance(width, int):
        raise TypeError("Expected 'width' to be an integer")
    if not isinstance(height, int):
        raise TypeError("Expected 'height' to be an integer")

    try:
        X, _ = create_coordinate_grids(width, height)
        return X
    except Exception as e:
        print(f"Exception: {e}")
        sys.exit(1)  # Terminate immediately

def create_linear_gradients(width: int = 256, height: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates two orthogonal linear gradients.
    Args:
        width (int): Width of the gradient in pixels. Default is 256.
        height (int): Height of the gradient in pixels. Default is 256.
    Returns:
        tuple: Two arrays representing X and Y linear gradients
    """

    if not isinstance(width, int):
        raise TypeError("Expected 'width' to be an integer")
    if not isinstance(height, int):
        raise TypeError("Expected 'height' to be an integer")

    try:
        X, Y = create_coordinate_grids(width, height)
        linear_x = X
        linear_y = Y
        return linear_x, linear_y
    except Exception as e:
        print(f"Exception: {e}")
        sys.exit(1)  # Terminate immediately

def create_radial_gradient(width: int = 256, height: int = 256, center_x: float = 0.5, center_y: float = 0.5) -> np.ndarray:
    """Creates a radial gradient centered at the specified coordinates.
    Args:
        width (int): Width of the gradient in pixels. Default is 256.
        height (int): Height of the gradient in pixels. Default is 256.
        center_x (float): X-coordinate for the center of the gradient. Default is 0.5.
        center_y (float): Y-coordinate for the center of the gradient. Default is 0.5.
    Returns:
        np.ndarray: Radial gradient array
    """

    if not isinstance(width, int):
        raise TypeError("Expected 'width' to be an integer")
    if not isinstance(height, int):
        raise TypeError("Expected 'height' to be an integer")
    if not isinstance(center_x, float):
        raise TypeError("Expected 'center_x' to be an float")
    if not isinstance(center_y, float):
        raise TypeError("Expected 'center_y' to be an float")

    try:
        X, Y = create_coordinate_grids(width, height)
        radial = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        radial = radial / np.max(radial)
        return radial
    except Exception as e:
        print(f"Exception: {e}")
        sys.exit(1)  # Terminate immediately

def apply_direction_to_gradient(gradient: np.ndarray, direction: Direction) -> np.ndarray:
    """
    Applies a direction to a gradient (normal or reversed).
    Args:
        gradient (np.ndarray): Input gradient array
        direction (Direction): Direction enum value to apply
    Returns:
        np.ndarray: Gradient with applied direction
    """

    if not isinstance(gradient, np.ndarray):
        raise TypeError("Expected 'gradient' to be an ndarray")
    if not isinstance(direction, Direction):
        raise TypeError("Expected 'direction' to be an Direction (declared enum)")

    try:
        if direction == Direction.REVERSE:
            return 1 - gradient
        return gradient
    except Exception as e:
        print(f"Exception: {e}")
        sys.exit(1)  # Terminate immediately

def create_directional_gradients(
    direction_x: Direction = Direction.FORWARD,
    direction_y: Direction = Direction.FORWARD,
    width: int = 256,
    height: int = 256
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates directional gradients with specified directions.
    Args:
        direction_x (Direction): Direction for X-axis gradient. Default is FORWARD.
        direction_y (Direction): Direction for Y-axis gradient. Default is FORWARD.
        width (int): Width of the gradient in pixels. Default is 256.
        height (int): Height of the gradient in pixels. Default is 256.
    Returns:
        tuple: Two arrays representing directional gradients
    """

    if not isinstance(width, int):
        raise TypeError("Expected 'width' to be an integer")
    if not isinstance(height, int):
        raise TypeError("Expected 'height' to be an integer")
    if not isinstance(direction_x, Direction):
        raise TypeError("Expected 'direction_x' to be an Direction (declared enum)")
    if not isinstance(direction_y, Direction):
        raise TypeError("Expected 'direction_x' to be an Direction (declared enum)")

    try:
        linear_x, linear_y = create_linear_gradients(width, height)
        linear_x = apply_direction_to_gradient(linear_x, direction_x)
        linear_y = apply_direction_to_gradient(linear_y, direction_y)
        return linear_x, linear_y
    except Exception as e:
        print(f"Exception: {e}")
        sys.exit(1)  # Terminate immediately

def create_rgb_combination(
    red_grad: np.ndarray,
    green_grad: np.ndarray,
    blue_grad: np.ndarray
) -> np.ndarray:
    """
    Combines three gradient channels into an RGB image.
    Args:
        red_grad (np.ndarray): Red channel gradient
        green_grad (np.ndarray): Green channel gradient
        blue_grad (np.ndarray): Blue channel gradient
    Returns:
        np.ndarray: 3D array representing RGB image
    """

    if not isinstance(red_grad, np.ndarray):
        raise TypeError("Expected 'red_grad' to be an ndarray")
    if not isinstance(green_grad, np.ndarray):
        raise TypeError("Expected 'green_grad' to be an ndarray")
    if not isinstance(blue_grad, np.ndarray):
        raise TypeError("Expected 'blue_grad' to be an ndarray")

    try:
        return np.stack([red_grad, green_grad, blue_grad], axis=2)
    except Exception as e:
        print(f"Exception: {e}")
        sys.exit(1)  # Terminate immediately

def visualize_gradient_channel(
    gradient: np.ndarray,
    cmap: str = 'viridis',
    title: str = '',
    ax: Optional[plt.Axes] = None
) -> None:
    """Visualizes a single-channel gradient using matplotlib.
    Args:
        gradient (np.ndarray): Gradient to visualize
        cmap (str): Colormap name. Default is 'viridis'.
        title (str): Plot title. Default is empty string.
        ax: Optional Axes object for plotting
    """

    if not isinstance(gradient, np.ndarray):
        raise TypeError("Expected 'gradient' to be an ndarray")
    if not isinstance(cmap, str):
        raise TypeError("Expected 'cmap' to be an str")
    if not isinstance(title, str):
        raise TypeError("Expected 'title' to be an str")

    try:
        if ax is None:
            plt.imshow(gradient, cmap=cmap)
            plt.title(title)
            plt.axis('off')
        else:
            if not isinstance(ax, Optional[plt.Axes]):
                raise TypeError("Expected 'ax' to be an Optional[plt.Axes]")

            ax.imshow(gradient, cmap=cmap)
            ax.set_title(title)
            ax.axis('off')
    except Exception as e:
        print(f"Exception: {e}")
        sys.exit(1)  # Terminate immediately

def save_gradient_image(
    gradient: np.ndarray,
    filename: str,
    cmap: str = 'viridis',
    dpi: int = 100,
    bbox_inches: str = 'tight') -> None:
    """Saves a single gradient image to a file.

    Args:
        gradient: 2D array of the gradient (values between 0 and 1)
        filename: Path to save the image file
        cmap: Colormap name (default: 'viridis')
        dpi: Dots per inch (default: 100)
        bbox_inches: How to handle figure margins (default: 'tight')
    """
    if not isinstance(gradient, np.ndarray):
        raise TypeError("Expected 'gradient' to be an ndarray")
    if not isinstance(cmap, str):
        raise TypeError("Expected 'cmap' to be an str")
    if not isinstance(filename, str):
        raise TypeError("Expected 'title' to be an str")
    if not isinstance(dpi, int):
        raise TypeError("Expected 'dpi' to be an int")
    if not isinstance(bbox_inches, str):
        raise TypeError("Expected 'bbox_inches' to be an str")

    try:
        plt.figure()
        plt.imshow(gradient, cmap=cmap)
        plt.axis('off')
        plt.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)
        plt.close()
    except Exception as e:
        print(f"Exception: {e}")
        sys.exit(1)  # Terminate immediately



# create visualisations
def create_linear_gradient_visualization() -> None:
    directions = [
        (Direction.FORWARD, Direction.FORWARD),
        (Direction.REVERSE, Direction.FORWARD),
        (Direction.FORWARD, Direction.REVERSE),
        (Direction.REVERSE, Direction.REVERSE)
    ]

    direction_names = [
        'Normal (Left→Right, Top→Bottom)',
        'Reverse X (Right→Left, Top→Bottom)',
        'Reverse Y (Left→Right, Bottom→Top)',
        'Both Reversed (Right→Left, Bottom→Top)'
    ]

    plt.figure(figsize=(16, 10))

    for i, (dir_x, dir_y) in enumerate(directions):
        linear_x, linear_y = create_directional_gradients(dir_x, dir_y)

        ax1 = plt.subplot(2, 4, i + 1)
        visualize_gradient_channel(linear_x, cmap='Reds', title=f'{direction_names[i]}\nX Gradient')

        ax2 = plt.subplot(2, 4, i + 5)
        visualize_gradient_channel(linear_y, cmap='Greens', title=f'{direction_names[i]}\nY Gradient')

    plt.tight_layout()
    plt.show()

def create_radial_gradient_visualization() -> None:
    radial = create_radial_gradient()
    plt.figure(figsize=(8, 6))
    visualize_gradient_channel(radial, cmap='Blues', title='Radial Gradient')
    plt.show()

def create_combined_gradients_visualization() -> None:
    combinations = [
        ("Linear X + Radial Y", "X Linear", "Y Radial"),
        ("Radial X + Linear Y", "X Radial", "Y Linear"),
        ("Linear X + Linear Y", "X Linear", "Y Linear")
    ]

    plt.figure(figsize=(16, 10))

    for i, (title, x_type, y_type) in enumerate(combinations):
        if x_type == "X Linear":
            linear_x, _ = create_linear_gradients()
        else:
            linear_x = create_radial_gradient()

        if y_type == "Y Linear":
            _, linear_y = create_linear_gradients()
        else:
            linear_y = create_radial_gradient()

        rgb_image = create_rgb_combination(linear_x, linear_y, linear_y)

        ax = plt.subplot(1, 3, i + 1)
        ax.imshow(rgb_image)
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def create_advanced_gradient_combinations() -> None:
    linear_x, linear_y = create_linear_gradients()
    radial = create_radial_gradient()

    combinations = [
        ("X Linear + Y Radial + Z Radial", linear_x, radial, radial),
        ("X Radial + Y Linear + Z Linear", radial, linear_y, linear_y),
        ("X Radial + Y Radial + Z Linear", radial, radial, linear_y)
    ]

    plt.figure(figsize=(16, 10))

    for i, (title, red_grad, green_grad, blue_grad) in enumerate(combinations):
        rgb_image = create_rgb_combination(red_grad, green_grad, blue_grad)

        ax = plt.subplot(1, 3, i + 1)
        ax.imshow(rgb_image)
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def main() -> None:
    print("Creating linear gradient visualizations...")
    create_linear_gradient_visualization()

    print("Creating radial gradient visualization...")
    create_radial_gradient_visualization()

    print("Creating combined gradients visualization...")
    create_combined_gradients_visualization()

    print("Creating advanced gradient combinations...")
    create_advanced_gradient_combinations()

    print("Creating linear white to black gradient visualization...")
    white_to_black = apply_direction_to_gradient(create_black_to_white_gradient(), Direction.REVERSE)
    visualize_gradient_channel(white_to_black, cmap='gray', title='White to Black')
    plt.show()

if __name__ == "__main__":
    main()
