import numpy as np

def izracun_kotov(b, p, l_1, l_2, h, psi_x_deg, psi_y_deg):
    # transformation from "in the x/y axis direction" to "around x/y axis"
    psi_y = np.deg2rad(psi_x_deg)
    psi_x = -np.deg2rad(psi_y_deg)

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

    En_1_m, En_1_p, En_2_m, En_2_p, En_3_m, En_3_p = 0,0,0,0,0,0
    if A_1**2+B_1**2-C_1**2 >= 0:
        En_1_m = 2*np.atan2(-B_1-np.sqrt(A_1**2+B_1**2-C_1**2), C_1-A_1)
        En_1_p = 2*np.atan2(-B_1+np.sqrt(A_1**2+B_1**2-C_1**2), C_1-A_1)
    if A_2**2+B_2**2-C_2**2 >= 0:
        En_2_p = 2*np.atan2(-B_2+np.sqrt(A_2**2+B_2**2-C_2**2), C_2-A_2)
        En_2_m = 2*np.atan2(-B_2-np.sqrt(A_2**2+B_2**2-C_2**2), C_2-A_2)
    if A_3**2+B_3**2-C_3**2 >= 0:
        En_3_p = 2*np.atan2(-B_3+np.sqrt(A_3**2+B_3**2-C_3**2), C_3-A_3)
        En_3_m = 2*np.atan2(-B_3-np.sqrt(A_3**2+B_3**2-C_3**2), C_3-A_3)

    #print(np.rad2deg(En_1_p), np.rad2deg(En_2_p), np.rad2deg(En_3_p))
    
    return np.array([np.rad2deg(En_1_p), np.rad2deg(En_2_p), np.rad2deg(En_3_p)])

#print(izracun_kotov(0.55, 0.275, 0.7, 0.775, 1.2, -0.2, 0.2))

def deg2steps(deg=np.ndarray):
        '''Converts degrees to motor steps by equation steps=deg*4/1.8*16/100'''
        if len(deg) != 3:
            raise(ValueError(f'Wrong input length \"{deg}\". Should be 3-list (x,y,z). x, y or z can be None.'))
        steps=np.array(-deg*4/1.8*16/100)

        return steps

def limit_steps(input_angles, min, max):
        if len(input_angles) != 3:
            raise(ValueError(f'Wrong input length \"{input_angles}\". Should be 3-list (x,y,z). x, y or z can be None.'))
     
        x,y,z = input_angles
        if x < min:
           x = min
        if y < min:
            y = min
        if z < min:
            z = min
        if x > max:
           x = max
        if y > max:
            y = max
        if z > max:
            z = max
        return np.array([x,y,z])

    


class PID_controller():
    '''
    Class for PID control. 


    Parameters:
    -start_position
    -target_position - on startup
    -max_tilt_angle 
    -max_change_rate
    -K_p
    -K_i
    -K_d
        
    
    '''
    def __init__(self, start_position, target_position, max_tilt_angle, max_change_rate, K_p, K_i, K_d):
        self.previous_error = (start_position-target_position) * np.array([1, -1]) # axis transform for right-handed coordinate frame
        self.integral = np.array([0, 0])
        self.max_tilt_angle = max_tilt_angle
        self.max_change_rate = max_change_rate
        self.K_p = K_p
        self.K_i = K_i
        self.K_d = K_d

        self.old_pos = start_position
         
    def calculate(self, current_position=np.ndarray, target_position=np.ndarray, time_step=np.float64):
        '''
        Calculate required angle to reach target position after one time step.
        
        '''
        self.current_error = (target_position-current_position) *  np.array([1, -1]) # axis transform for right-handed coordinate frame
        #self.adjust_PID()
        proportional = self.K_p*self.current_error
        self.integral = self.integral + self.current_error*time_step
        #self.integral = np.clip(self.integral, -100, 100)

        integral_term = self.K_i*self.integral

        # pristop z D term za pozicijo, ne error
        #derivative = (error-self.previous_error)/time_step
        derivative = -(current_position-self.old_pos)/time_step * np.array([1, -1])
        derivative_term = self.K_d*derivative

        new_tilt_angle = proportional+integral_term+derivative_term 

        self.previous_error = self.current_error
        self.old_pos = current_position

        return new_tilt_angle
    
    def clip_angle(self, tilt_angle, new_tilt_angle, time_step):
        '''
        Clip calculated angle not to exceed hardware maximum. 
        
        '''
        
        tilt_angle_change = new_tilt_angle - tilt_angle
        max_change = self.max_change_rate * time_step
        tilt_angle_change_clipped = np.clip(tilt_angle_change, -max_change, max_change)

        tilt_angle_clipped = np.clip(tilt_angle + tilt_angle_change_clipped, -self.max_tilt_angle, self.max_tilt_angle)
        return tilt_angle_clipped
    
    def adjust_PID(self):
        x = min(150, np.linalg.norm(self.current_error))
        self.K_p = 0.00025 - (6.6666667e-7)*x
        self.K_i = -0.0000291363 + 0.000479136*(np.e)**(-0.0184417*x)
        self.K_d = 0.0005 - (2.8e-6)*x