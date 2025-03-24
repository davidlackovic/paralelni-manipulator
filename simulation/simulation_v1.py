import numpy as np
import pygame
import sys
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


pygame.init()

WIDTH, HEIGHT = 800, 600
BALL_RADIUS = 10
BALL_COLOR = (0, 0, 255)
PLATE_COLOR = (150, 150, 150)
BACKGROUND_COLOR = (200, 200, 200)
FPS = 200
g = 9.81

target_position = np.array([WIDTH/2, HEIGHT/2])     #ciljna lega
tilt_angle = np.array([0, 0])                       #zacetni naklon
time_step = 30/FPS

max_change_rate = np.deg2rad(5) #deg/s
max_tilt_angle = np.deg2rad(18) #deg

#PID konstante
K_p = 0.001
K_i = 0.00
K_d = 0.01



screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Simulacija")

clock = pygame.time.Clock()

p_0 = np.array([400, 300]) #zacetna lega
v_0 = np.array([1, 3]) #zacetna hitrost

#PID
previous_error = p_0-target_position
integral = np.array([0, 0])

def equations_of_motion(t, y, angle):
    p, v = y
    dpdt = v
    dvdt = g*np.sin(angle)
    return [dpdt, dvdt]


def integrate_motion(p_0 = np.ndarray, v_0=np.ndarray, angle=np.ndarray, t_end=float):
    sol_x = solve_ivp(equations_of_motion, [0, t_end], [p_0[0], v_0[0]], args=(angle[0],), t_eval=[t_end])
    sol_y = solve_ivp(equations_of_motion, [0, t_end], [p_0[1], v_0[1]], args=(angle[1],), t_eval=[t_end])
    
    return np.array([sol_x.y[0][-1], sol_y.y[0][-1]]), np.array([sol_x.y[1][-1], sol_y.y[1][-1]]) 

def pid_controller(current_postition=np.ndarray):
    global previous_error, integral, tilt_angle

    error = target_position-current_postition

    proportional = K_p*error

    integral = integral + error*time_step

    integral_term = K_i*integral

    derivative = (error-previous_error)/time_step
    derivative_term = K_d*derivative

    new_tilt_angle = proportional+integral_term+derivative_term 
    
    tilt_angle_change = new_tilt_angle - tilt_angle
    max_change = max_change_rate * time_step
    tilt_angle_change_clamped = np.clip(tilt_angle_change, -max_change, max_change)

    tilt_angle = np.clip(tilt_angle + tilt_angle_change_clamped, -max_tilt_angle, max_tilt_angle)

    print(new_tilt_angle, tilt_angle_change_clamped, tilt_angle_change, np.rad2deg(tilt_angle))

    previous_error = error

def draw_plate():
    pygame.draw.rect(screen, PLATE_COLOR, (0, 0, WIDTH, HEIGHT))
    pygame.draw.circle(screen, (255,0,0), (int(WIDTH/2), int(HEIGHT/2)), 50, 1)
    pygame.draw.circle(screen, (255,0,0), (int(WIDTH/2), int(HEIGHT/2)), 100, 1)


def draw_ball(coordinates=np.ndarray):
    pygame.draw.circle(screen, BALL_COLOR, (int(coordinates[0]), int(coordinates[1])), BALL_RADIUS)

def draw_tilt_line(tilt_angle):
    pygame.draw.line(screen, (0,0,0), (int(WIDTH/2), int(HEIGHT/2)), (int(WIDTH/2+10*np.rad2deg(tilt_angle[0])), int(HEIGHT/2+10*np.rad2deg(tilt_angle[1]))), 3)
    pygame.draw.circle(screen, (0,0,0), (int(WIDTH/2), int(HEIGHT/2)), 4)


def main():
     global p_0, v_0, tilt_angle, target_position

     while True:
        #print(p_0, v_0, tilt_angle)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mouse_pos = pygame.mouse.get_pos()
            target_position = np.array([mouse_pos[0], mouse_pos[1]])

    
        pid_controller(p_0)

        p_0, v_0 = integrate_motion(p_0, v_0, tilt_angle, time_step)

        screen.fill(BACKGROUND_COLOR)
        draw_plate()
        draw_ball(p_0)
        draw_tilt_line(tilt_angle=tilt_angle)

        pygame.display.flip()
        clock.tick(FPS)

if __name__ == "__main__":
    main()

