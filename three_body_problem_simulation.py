import numpy
import matplotlib.pyplot as plt
import math
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

def three_body_problem():
    # three indices: which star, xyz
    positions = numpy.zeros([3, 3])  # m
    velocities = numpy.zeros([3, 3])  # m / s
    positions_euler = numpy.zeros([3, 3])  # m
    velocities_euler = numpy.zeros([3, 3])  # m / s
    positions_heun = numpy.zeros([3, 3])  # m
    velocities_heun = numpy.zeros([3, 3])  # m / s
    positions[0] = numpy.array([-1., 2., 2.]) * 1e11
    positions[1] = numpy.array([3., -2., 1.]) * 1e11
    positions[2] = numpy.array([2., 3., -1.]) * 1e11
    velocities[0]=numpy.array([-2., 0.5, 5.]) * 1e3
    velocities[1]=numpy.array([7., 0.5, 2.]) * 1e3
    velocities[2]=numpy.array([-4., -0.5, -3.])*1e3
    # used to append the positions
    pos1=[]
    pos1.append(positions[0].tolist())
    pos2=[]
    pos2.append(positions[1].tolist())
    pos3=[]
    pos3.append(positions[2].tolist())
    start_time = 0.  # s
    end_time = 50. * 365.26 * 24. * 3600.  # s
    h = 2. * 3600.  # s
    tolerance = 5e5  # m
    # use the adaptive step size instead of fixed step size
    while start_time < end_time:
        # Implement the symplectic Euler Method for the motion
        positions_euler[0] = positions[0] + h * velocities[0]
        positions_euler[1] = positions[1] + h * velocities[1]
        positions_euler[2] = positions[2] + h * velocities[2]
        velocities_euler[0] = velocities[0] + h * acceleration1(positions[0], positions[1], positions[2])
        velocities_euler[1] = velocities[1] + h * acceleration2(positions[1], positions[0], positions[2])
        velocities_euler[2] = velocities[2] + h * acceleration3(positions[2], positions[0], positions[1])
        # Implement the Heun's mothod
        positions_heun[0]=positions[0]+h*0.5*(velocities[0]+velocities_euler[0])
        positions_heun[1]=positions[1]+h*0.5*(velocities[1]+velocities_euler[1])
        positions_heun[2]=positions[2]+h*0.5*(velocities[2]+velocities_euler[2])
        velocities_heun[0]=velocities[0]+h*0.5*(acceleration1(positions[0], positions[1], positions[2])
                                                +acceleration1(positions_euler[0], positions_euler[1], positions_euler[2]))
        velocities_heun[1]=velocities[1]+h*0.5*(acceleration2(positions[1], positions[0], positions[2])
                                                +acceleration2(positions_euler[1], positions_euler[0], positions_euler[2]))
        velocities_heun[2]=velocities[2]+h*0.5*(acceleration3(positions[2], positions[0], positions[1])
                                                +acceleration3(positions_euler[2], positions_euler[0], positions_euler[1]))
        positions[0]=positions_heun[0]
        positions[1]=positions_heun[1]
        positions[2]=positions_heun[2]
        velocities[0]=velocities_heun[0]
        velocities[1]=velocities_heun[1]
        velocities[2]=velocities_heun[2]
        error1=numpy.linalg.norm(positions_euler[0]-positions_heun[0])+end_time*numpy.linalg.norm(velocities_euler[0]-velocities_heun[0])
        error2=numpy.linalg.norm(positions_euler[1]-positions_heun[1])+end_time*numpy.linalg.norm(velocities_euler[1]-velocities_heun[1])
        error3=numpy.linalg.norm(positions_euler[2]-positions_heun[2])+end_time*numpy.linalg.norm(velocities_euler[2]-velocities_heun[2])
        h_new1=h*math.sqrt(tolerance/error1)
        h_new2=h*math.sqrt(tolerance/error2)
        h_new3=h*math.sqrt(tolerance/error3)
        # use the average value to update
        h_new= (h_new1+h_new2+h_new3)/3.
        pos1.append(positions[0].tolist())
        pos2.append(positions[1].tolist())
        pos3.append(positions[2].tolist())
        start_time+=h
        h=h_new
    return pos1, pos2, pos3

pos1, pos2, pos3 = three_body_problem()
# pprint.pprint(pos1)
# print ('----------------------------')
# pprint.pprint(pos2)
# print ('----------------------------')
# pprint.pprint(pos3)

def plot_stars():
    plt.ion()
    plt.show()
    fig=plt.figure()
    axes=Axes3D(fig)
    #axes = matplotlib.pyplot.figure().add_subplot(111, projection='3d')
    axes.axis('equal')
    axes.set_xlabel('x in m')
    axes.set_ylabel('y in m')
    axes.set_zlabel('z in m')
    #axes.set_xlim((0, 0))
    #axes.set_ylim((0, 0))
    #axes.set_zlim((0, 0))
    for i in range(20000):    # len(pos1)
        axes.scatter(pos1[i][0], pos1[i][1], pos1[i][2], c='r', marker='o', norm=0.7)
        axes.scatter(pos2[i][0], pos2[i][1], pos2[i][2], c='b', marker='o', norm=0.8)
        axes.scatter(pos3[i][0], pos3[i][1], pos3[i][2], c='y', marker='o', norm=0.9)
        plt.draw()
        plt.pause(0.001)

plot_stars()