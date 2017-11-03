import numpy
import matplotlib.pyplot
from mpl_toolkits.mplot3d import Axes3D

star_1_mass = 1e30  # kg
star_2_mass = 2e30  # kg
star_3_mass = 3e30  # kg
gravitational_constant = 6.67e-11  # m3 / kg s2


def acceleration1(star1_position, star2_position, star3_position):
    vector_from_star1_to_star2 = - star1_position + star2_position
    vector_from_star1_to_star3 = - star1_position + star3_position
    acceleration_star1 = gravitational_constant * (star_2_mass / numpy.linalg.norm(
        vector_from_star1_to_star2) ** 3 * vector_from_star1_to_star2 + star_3_mass / numpy.linalg.norm(
        vector_from_star1_to_star3) ** 3 * vector_from_star1_to_star3)
    return acceleration_star1


def acceleration2(star2_position, star1_position, star3_position):
    vector_from_star2_to_star1 = - star2_position + star1_position
    vector_from_star2_to_star3 = - star2_position + star3_position
    acceleration_star2 = gravitational_constant * (star_1_mass / numpy.linalg.norm(
        vector_from_star2_to_star1) ** 3 * vector_from_star2_to_star1 + star_3_mass / numpy.linalg.norm(
        vector_from_star2_to_star3) ** 3 * vector_from_star2_to_star3)
    return acceleration_star2


def acceleration3(star3_position, star1_position, star2_position):
    vector_from_star3_to_star1 = - star3_position + star1_position
    vector_from_star3_to_star2 = - star3_position + star2_position
    acceleration_star3 = gravitational_constant * (star_1_mass / numpy.linalg.norm(
        vector_from_star3_to_star1) ** 3 * vector_from_star3_to_star1 + star_2_mass / numpy.linalg.norm(
        vector_from_star3_to_star2) ** 3 * vector_from_star3_to_star2)
    return acceleration_star3


end_time = 50. * 365.26 * 24. * 3600.  # s
h = 2. * 3600.  # s
num_steps = int(end_time / h)
times = h * numpy.array(range(num_steps + 1))


def three_body_problem():
    # three indices: which time, which star, xyz
    positions = numpy.zeros([num_steps + 1, 3, 3])  # m
    velocities = numpy.zeros([num_steps + 1, 3, 3])  # m / s
    positions[0] = numpy.array([[-1.1, 3., 2.], [6., -5., 4.], [7., 8., -7.]]) * 1e11
    velocities[0] = numpy.array([[-2., 0.5, 5.], [7., 0.5, 2.], [-4., -0.5, -3.]]) * 1e3

    for step in range(num_steps):
        # Implement the symplectic Euler Method for the motion
        positions[step + 1][0] = positions[step][0] + h * velocities[step][0]
        positions[step + 1][1] = positions[step][1] + h * velocities[step][1]
        positions[step + 1][2] = positions[step][2] + h * velocities[step][2]
        velocities[step + 1][0] = velocities[step][0] + h * acceleration1(positions[step + 1][0],
                                                                          positions[step + 1][1],
                                                                          positions[step + 1][2])
        velocities[step + 1][1] = velocities[step][1] + h * acceleration2(positions[step + 1][1],
                                                                          positions[step + 1][0],
                                                                          positions[step + 1][2])
        velocities[step + 1][2] = velocities[step][2] + h * acceleration3(positions[step + 1][2],
                                                                          positions[step + 1][0],
                                                                          positions[step + 1][1])

    return positions, velocities


positions, velocities = three_body_problem()

def plot_stars():
    axes = matplotlib.pyplot.gca()
    axes.set_xlabel('x in m')
    axes.set_ylabel('z in m')
    axes.plot(positions[:, 0, 0], positions[:, 0, 2])
    axes.plot(positions[:, 1, 0], positions[:, 1, 2])
    axes.plot(positions[:, 2, 0], positions[:, 2, 2])
    matplotlib.pyplot.axis('equal')
    matplotlib.pyplot.title('Three Body Problem Simulation (Trajectories of three star system)')
    matplotlib.pyplot.show()

plot_stars()
