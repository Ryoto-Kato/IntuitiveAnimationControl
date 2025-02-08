import numpy as np
from pytransform3d.transformations import pq_slerp
from pytransform3d.rotations import matrix_from_axis_angle, quaternion_from_matrix, quaternion_slerp, matrix_from_quaternion

# def quaterion_interpolation(start_R, end_R, num_steps):
#     start_q = quaternion_from_matrix(start_R)
#     end_q = quaternion_from_matrix(end_R)

#     interp_Rs = []
#     T = np.linspace(0, 1, num_steps)
#     for i, t in enumerate(T):
#         _inter_q = quaternion_slerp(start_q, end_q, t)
#         _inter_R = quaternion_from_matrix(_inter_q)
#         interp_Rs.append(_inter_R)

def transform_interpolation(start_R, end_R, start_t, end_t, time_step):
    start_q = quaternion_from_matrix(start_R)
    end_q = quaternion_from_matrix(end_R)

    start_qt = np.hstack((start_t, start_q))
    end_qt = np.hstack((end_t, end_q))

    # print("start:", start_qt)
    inter_qt = pq_slerp(start = start_qt, end = end_qt, t = time_step)
    # print("time t:", inter_qt)
    # print("end:", start_qt)

    inter_R = matrix_from_quaternion(inter_qt[3:])
    inter_t = inter_qt[:3]
    # print(inter_qt)
    # print(inter_R,inter_t)

    return inter_R, inter_t
    

    







