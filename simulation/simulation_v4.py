import numpy as np
import pygame
import sys
import os

main_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if main_folder not in sys.path:
     sys.path.append(main_folder)

import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time
import simulation_library
from finished import kinematika
WIDTH = 640
HEIGHT = 480

p_0 = np.array([0, 0]) #zacetna lega
v_0 = np.array([0, 0]) #zacetna hitrost
p_0_screen = np.array([p_0[0]+WIDTH//2, -p_0[1]+HEIGHT//2])


target_position = np.array([0, 0])                              #ciljna lega
tilt_angle = np.deg2rad(np.array([0, 0]))                       #zacetni naklon

time_step = 0.07
time_delta = 0.035 # za PID

max_change_rate = np.deg2rad(5) #deg/s
max_tilt_angle = np.deg2rad(18) #deg

K_p = 0.0008
K_i = 0.00000
K_d = 0.01

#K_p = 0.0005
#K_i = 0.0005
#K_d = 0.0005


SIM = simulation_library.SimulationGame(p_0, target_position, WIDTH, HEIGHT)
PID = kinematika.PID_controller(p_0, target_position, max_tilt_angle, max_change_rate, K_p, K_i, K_d)


old_time = time.time()
while True:
        #print(p_0, v_0, tilt_angle)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mouse_pos = pygame.mouse.get_pos()
            target_position = np.array([mouse_pos[0]-WIDTH//2, -mouse_pos[1]+HEIGHT//2])

        p_0, v_0 = SIM.integrate_motion(p_0, v_0, tilt_angle , time_step)
        #print(f'[{time.strftime('%Y-%m-%d %H:%M:%S')}.{int(time.time() * 1000) % 1000:03d}]', f'simulation: {p_0, tilt_angle}')
        


        x,y = p_0
        vx, vy = v_0
        noise_value = 0.0
        x_rand = x + np.random.rand(1)[0]*noise_value
        y_rand = y + np.random.rand(1)[0]*noise_value
        loss_factor = 0.99
        vx_rand = vx * loss_factor
        vy_rand = vy * loss_factor
        p_0_noise = np.array([x_rand, y_rand])
        v_0_noise = np.array([vx_rand, vy_rand])
        old_time=time.time()  
        old_tilt = tilt_angle 
        tilt_angle = PID.calculate(p_0, target_position, time_delta) * np.array([1, -1]) 
        tilt_angle = PID.clip_angle(tilt_angle, old_tilt, time_step)
        print(p_0, p_0_noise, target_position, tilt_angle, p_0_screen)

        p_0_screen = np.array([p_0[0]+WIDTH//2, -p_0[1]+HEIGHT//2])
        SIM.screen.fill(SIM.BACKGROUND_COLOR)
        SIM.draw_plate()
        SIM.draw_ball(p_0_screen)
        SIM.draw_tilt_line(tilt_angle=tilt_angle* np.array([1, -1]) )
        

        pygame.display.flip()
        #clock.tick(FPS)