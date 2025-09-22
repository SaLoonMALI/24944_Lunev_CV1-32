"""
 ============================================================================
 Name        : create_gradient_image.py
 Author      : Sam Lunev
 Version     : .01
 Copyright   : All rights reserved
 Description : Block 1 CV-32 task
 ============================================================================
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Union, Any
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

def validate_width_and_height(width: Union[int, float], height: Union[int, float]) -> None:
    """
    Validate 'width' and 'height' parameters are positive integers.

    Args:
        width: Width parameter to validate
        height: Height parameter to validate

    Raises:
        TypeError: If parameters are not numeric
        ValueError: If parameters are not positive
    """
    if not isinstance(width, (int, float)):
        raise TypeError("Expected 'width' to be numeric")
    if not isinstance(height, (int, float)):
        raise TypeError("Expected 'height' to be numeric")
    if width <= 0 or height <= 0:
        raise ValueError("Width and height must be positive")

def validate_center_coordinates(center_x: Union[int, float], center_y: Union[int, float]) -> None:
    """
    Validate center coordinates for radial gradients.

    Args:
        center_x: X coordinate of center
        center_y: Y coordinate of center

    Raises:
        TypeError: If parameters are not numeric
    """
    if not isinstance(center_x, (int, float)):
        raise TypeError("Expected 'center_x' to be numeric")
    if not isinstance(center_y, (int, float)):
        raise TypeError("Expected 'center_y' to be numeric")

def validate_direction(direction: Direction) -> None:
    """
    Validate gradient direction parameter.

    Args:
        direction: Direction to validate

    Raises:
        TypeError: If parameter is not a Direction enum
    """
    if not isinstance(direction, Direction):
        raise TypeError("Expected 'direction' to be Direction enum")

def validate_gradient_array(gradient: np.ndarray) -> None:
    """
    Validate gradient array.

    Args:
        gradient: Gradient array to validate

    Raises:
        TypeError: If parameter is not a numpy array
    """
    if not isinstance(gradient, np.ndarray):
        raise TypeError("Expected 'gradient' to be numpy array")

def create_coordinate_grids(width: int = 256, height: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create coordinate grids for gradient calculations.

    Args:
        width (int): Width of the grid
        height (int): Height of the grid

    Returns:
        Tuple[np.ndarray, np.ndarray]: X and Y coordinate arrays
    """
    validate_width_and_height(width, height)

    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    X, Y = np.meshgrid(x, y)
    return X, Y

def create_black_to_white_gradient(width: int = 256, height: int = 256) -> np.ndarray:
    """
    Create a black to white linear gradient.

    Args:
        width (int): Width of the gradient
        height (int): Height of the gradient

    Returns:
        np.ndarray: Gradient array
    """
    validate_width_and_height(width, height)

    X, _ = create_coordinate_grids(width, height)
    return X

def create_linear_gradients(width: int = 256, height: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create linear gradients in x and y directions.

    Args:
        width (int): Width of the gradient
        height (int): Height of the gradient

    Returns:
        Tuple[np.ndarray, np.ndarray]: X and Y gradient arrays
    """
    validate_width_and_height(width, height)

    X, Y = create_coordinate_grids(width, height)
    return X, Y

def create_radial_gradient(width: int = 256, height: int = 256,
                          center_x: float = 0.5, center_y: float = 0.5) -> np.ndarray:
    """
    Create a radial gradient centered at specified coordinates.

    Args:
        width (int): Width of the gradient
        height (int): Height of the gradient
        center_x (float): X coordinate of center
        center_y (float): Y coordinate of center

    Returns:
        np.ndarray: Gradient array
    """
    validate_width_and_height(width, height)
    validate_center_coordinates(center_x, center_y)

    try:
        X, Y = create_coordinate_grids(width, height)
        radial = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        if np.max(radial) > 0:
            radial = radial / np.max(radial)
        return radial
    except Exception as e:
        raise RuntimeError(f"Error creating radial gradient: {e}")

def apply_direction_to_gradient(gradient: np.ndarray, direction: Direction) -> np.ndarray:
    """
    Apply direction to a gradient.

    Args:
        gradient (np.ndarray): Gradient array to modify
        direction (Direction): Direction to apply

    Returns:
        np.ndarray: Modified gradient array
    """
    validate_gradient_array(gradient)
    validate_direction(direction)

    if direction == Direction.REVERSE:
        return 1 - gradient
    else:
        return gradient

def create_rgb_combination(red: np.ndarray, green: np.ndarray, blue: np.ndarray) -> np.ndarray:
    """
    Combine red, green, and blue gradients into an RGB image.

    Args:
        red (np.ndarray): Red channel gradient
        green (np.ndarray): Green channel gradient
        blue (np.ndarray): Blue channel gradient

    Returns:
        np.ndarray: Combined RGB image array
    """
    validate_gradient_array(red)
    validate_gradient_array(green)
    validate_gradient_array(blue)

    try:
        rgb_image = np.stack([red, green, blue], axis=-1)
        return np.clip(rgb_image, 0, 1)
    except Exception as e:
        raise RuntimeError(f"Error combining RGB channels: {e}")

def visualize_gradient(gradient: np.ndarray, title: str = "Gradient",
                      cmap: str = "gray", return_image: bool = False) -> Optional[np.ndarray]:
    """
    Visualize a gradient with specified colormap.

    Args:
        gradient (np.ndarray): Gradient array to visualize
        title (str): Title for the plot
        cmap (str): Colormap to use
        return_image (bool): Whether to return the image array

    Returns:
        Optional[np.ndarray]: Gradient image array if return_image=True, else None
    """
    validate_gradient_array(gradient)

    try:
        plt.figure(figsize=(8, 6))
        plt.imshow(gradient, cmap=cmap)
        plt.title(title)
        plt.axis('off')
        plt.show()

        # Return the image data if requested
        if return_image:
            return gradient.copy()
        return None
    except Exception as e:
        raise RuntimeError(f"Error visualizing gradient: {e}")

def visualize_rgb_gradient(rgb_image: np.ndarray, title: str = "RGB Gradient",
                          return_image: bool = False) -> Optional[np.ndarray]:
    """
    Visualize an RGB gradient.

    Args:
        rgb_image (np.ndarray): RGB image array to visualize
        title (str): Title for the plot
        return_image (bool): Whether to return the image array

    Returns:
        Optional[np.ndarray]: RGB image array if return_image=True, else None
    """
    validate_gradient_array(rgb_image)

    try:
        plt.figure(figsize=(8, 6))
        plt.imshow(rgb_image)
        plt.title(title)
        plt.axis('off')
        plt.show()

        # Return the image data if requested
        if return_image:
            return rgb_image.copy()
        return None
    except Exception as e:
        raise RuntimeError(f"Error visualizing RGB gradient: {e}")

def create_gradient_samples() -> None:
    """
    Create sample gradient visualizations.

    Returns:
        None: Displays sample plots
    """
    try:
        # Linear gradient sample
        print("Creating linear gradient samples...")

        # Red channel (linear x-gradient)
        red, _ = create_linear_gradients()
        visualize_gradient(red, title="Red Gradient", cmap="Reds")

        # Green channel (linear y-gradient)
        _, green = create_linear_gradients()
        visualize_gradient(green, title="Green Gradient", cmap="Greens")

        # Blue channel (radial gradient)
        blue = create_radial_gradient()
        visualize_gradient(blue, title="Blue Gradient", cmap="Blues")

        # Combined RGB sample
        print("Creating combined RGB gradient...")
        red, green= create_linear_gradients()
        blue = create_radial_gradient()
        rgb_image = create_rgb_combination(red, green, blue)
        visualize_rgb_gradient(rgb_image, title="Combined RGB Gradient")

    except Exception as e:
        raise RuntimeError(f"Error creating gradient samples: {e}")

def main() -> None:
    """
    Main function to demonstrate gradient creation and visualization.

    Returns:
        None
    """
    try:
        # Create samples with return functionality
        print("Creating gradients with return capability...")

        # Create a gradient and return it
        red_gradient = create_linear_gradients()[0]
        returned_red = visualize_gradient(red_gradient, title="Red Gradient (Returned)",
                                        cmap="Reds", return_image=True)

        # Create RGB image and return it
        red, green = create_linear_gradients()
        blue = create_radial_gradient()
        rgb_image = create_rgb_combination(red, green, blue)
        returned_rgb = visualize_rgb_gradient(rgb_image, title="RGB Gradient (Returned)",
                                            return_image=True)

        # Normal usage without return
        print("\nCreating normal visualizations...")
        create_gradient_samples()
        
        print("Creating linear white to black gradient visualization...")
        white_to_black = apply_direction_to_gradient(create_black_to_white_gradient(), Direction.REVERSE)
        visualize_gradient(white_to_black, cmap='gray', title='White to Black')
        plt.show()

        print("All gradients created successfully!")

    except Exception as e:
        raise RuntimeError(f"Error in main function: {e}")

if __name__ == "__main__":
    main()
