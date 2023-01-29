import math

def calculate_density_and_youngs_modulus(mass, length, width, thickness, deflection):
    """
    Calculates the density and Young's modulus of a rod based on its mass, dimensions, and deflection due to self-weight.

    Parameters:
    - mass (float): Mass of the rod in kilograms.
    - length (float): Length of the rod in meters.
    - width (float): Width of the rod in meters.
    - thickness (float): Thickness of the rod in meters.
    - deflection (float): Deflection of the rod due to self-weight in meters.

    Returns:
    - A tuple containing the density and Young's modulus of the rod.
    """
    # Check if input values are valid
    if not (isinstance(mass, (int, float)) and mass > 0):
        raise ValueError("'mass' must be a positive number.")
    if not (isinstance(length, (int, float)) and length > 0):
        raise ValueError("'length' must be a positive number.")
    if not (isinstance(width, (int, float)) and width > 0):
        raise ValueError("'width' must be a positive number.")
    if not (isinstance(thickness, (int, float)) and thickness > 0):
        raise ValueError("'thickness' must be a positive number.")
    if not (isinstance(deflection, (int, float)) and deflection > 0):
        raise ValueError("'deflection' must be a positive number.")

    # Calculate density
    density = mass / (width * thickness * length)

    # Calculate moment of inertia
    moment_of_inertia = width * thickness ** 3 / 12

    # Calculate Young's modulus
    youngs_modulus = 5 * mass * 9.81 * length ** 3 / (384 * moment_of_inertia * deflection)

    return density, youngs_modulus

ans = calculate_density_and_youngs_modulus(1.65e-3, 10e-2, 1.5e-2, 0.9e-3, 0.919e-2)

print(ans)
































# import math
# import numpy as np
#
# g = 9.81 # acceleration due to gravity
# l = 10e-2 # length of the rod
# w = 1.5e-2 # width of the rod
#
# def calculateY(g, l, w, m, t, dy):
#     # m = 1.65e-3 # mass of the rod
#     # g = 9.81 # acceleration due to gravity
#     # l = 10e-2 # length of the rod
#     # w = 1.5e-2 # width of the rod
#     # t = 0.9e-3 # thickness of the rod
#     # dy = 0.919e-2 # deflection due to self-weight
#     rho = m / (w * t * l) # density of the rod
#     I = w * t ** 3 / 12 # moment of inertia of the rod
#     Y = 5 * m * g * l ** 3 / (384 * I * dy)  # Young's modulus
#     return rho, Y
#
#
# ans = calculateY(g, l, w, m = 1.65e-3, t = 0.9e-3, dy = 0.919e-2)
#
# print(ans)