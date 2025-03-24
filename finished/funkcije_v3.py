import numpy as np

def izracun_kotov(b, p, l_1, l_2, h, psi_x_deg, psi_y_deg):
    psi_x = np.deg2rad(psi_x_deg)
    psi_y = np.deg2rad(psi_y_deg)
    if np.sin(psi_y) == 1 or np.sin(psi_y) == -1:
        raise ValueError('Nepravilna vrednost omega_x!')
    
    #omega_x = np.sin(psi_y)
    #omega_y = -np.sin(psi_x)*np.cos(psi_y)
    psi_z = np.atan(-np.sin(psi_x)*np.sin(psi_y)/(np.cos(psi_x)+np.cos(psi_y)))

    R = np.array([[np.cos(psi_y)*np.cos(psi_z), -np.cos(psi_y)*np.sin(psi_z), np.sin(psi_y)],
                  [np.sin(psi_x)*np.sin(psi_y)*np.cos(psi_z)+np.cos(psi_x)*np.sin(psi_z), np.cos(psi_x)*np.cos(psi_z)-np.sin(psi_x)*np.sin(psi_y)*np.sin(psi_z), -np.sin(psi_x)*np.cos(psi_y)],
                  [np.sin(psi_x)*np.sin(psi_z)-np.cos(psi_x)*np.sin(psi_y)*np.cos(psi_z), np.sin(psi_x)*np.cos(psi_z)+np.cos(psi_x)*np.sin(psi_y)*np.sin(psi_z), np.cos(psi_x)*np.cos(psi_y)]])

    O_7 = np.array([p*(R[0][0]-R[1][1])/2, -R[1][0]*p, h])

    alpha = 0
    R_z = np.array([[np.cos(alpha), -np.sin(alpha), 0],
                    [np.sin(alpha), np.cos(alpha), 0],
                    [0, 0, 1]])
    vec_0 = O_7 + R@R_z@np.array([p, 0, 0])
    alpha = 2*np.pi/3 
    R_z = np.array([[np.cos(alpha), -np.sin(alpha), 0],
                    [np.sin(alpha), np.cos(alpha), 0],
                    [0, 0, 1]])
    vec_1 = O_7 + R@R_z@np.array([p, 0, 0])
    alpha = 4*np.pi/3 
    R_z = np.array([[np.cos(alpha), -np.sin(alpha), 0],
                    [np.sin(alpha), np.cos(alpha), 0],
                    [0, 0, 1]])
    vec_2 = O_7 + R@R_z@np.array([p, 0, 0])

    vecs = np.array([[vec_0[0], vec_0[2]], [vec_1[0], vec_1[2]], [vec_2[0], vec_2[2]]])

    A_1 = 2*l_1*np.cos(0)*(-vecs[0][0] + b*np.cos(0))
    A_2 = 2*l_1*np.cos(2*np.pi/3)*(-vecs[1][0] + b*np.cos(2*np.pi/3))
    A_3 = 2*l_1*np.cos(4*np.pi/3)*(-vecs[2][0] + b*np.cos(4*np.pi/3))

    B_1 = 2*l_1*vecs[0][1]*np.cos(0)**2
    B_2 = 2*l_1*vecs[1][1]*np.cos(2*np.pi/3)**2
    B_3 = 2*l_1*vecs[2][1]*np.cos(4*np.pi/3)**2

    C_1 = vecs[0][0]**2-2*b*vecs[0][0]*np.cos(0)+np.cos(0)**2*(b**2+l_1**2-l_2**2+vecs[0][1]**2)
    C_2 = vecs[1][0]**2-2*b*vecs[1][0]*np.cos(2*np.pi/3)+np.cos(2*np.pi/3)**2*(b**2+l_1**2-l_2**2+vecs[1][1]**2)
    C_3 = vecs[2][0]**2-2*b*vecs[2][0]*np.cos(4*np.pi/3)+np.cos(4*np.pi/3)**2*(b**2+l_1**2-l_2**2+vecs[2][1]**2)

    En_1_m = 2*np.atan2(-B_1-np.sqrt(A_1**2+B_1**2-C_1**2), C_1-A_1)
    En_1_p = 2*np.atan2(-B_1+np.sqrt(A_1**2+B_1**2-C_1**2), C_1-A_1)
    En_2_p = 2*np.atan2(-B_2+np.sqrt(A_2**2+B_2**2-C_2**2), C_2-A_2)
    En_2_m = 2*np.atan2(-B_2-np.sqrt(A_2**2+B_2**2-C_2**2), C_2-A_2)
    En_3_p = 2*np.atan2(-B_3+np.sqrt(A_3**2+B_3**2-C_3**2), C_3-A_3)
    En_3_m = 2*np.atan2(-B_3-np.sqrt(A_3**2+B_3**2-C_3**2), C_3-A_3)

    print(np.rad2deg(En_1_p), np.rad2deg(En_2_p), np.rad2deg(En_3_p))

    return np.rad2deg(En_1_p), np.rad2deg(En_2_p), np.rad2deg(En_3_p)

#print(izracun_kotov(0.55, 0.275, 0.7, 0.775, 1.2, -0.2, 0.2))