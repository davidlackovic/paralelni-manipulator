import numpy as np
import pygame
import sys
from scipy.integrate import solve_ivp
import time

class SimulationGame():
    def __init__(self, p_0, target_position, WIDTH, HEIGHT):
        pygame.init()
        self.p_0 = p_0
        self.target_position = target_position
        self.screen = pygame.display.set_mode((640, 480))
        pygame.display.set_caption("Simulacija")

        self.WIDTH, self.HEIGHT = WIDTH, HEIGHT
        self.BALL_RADIUS = 10
        self.BALL_COLOR = (0, 0, 255)
        self.PLATE_COLOR = (150, 150, 150)
        self.BACKGROUND_COLOR = (200, 200, 200)


    def equations_of_motion(self, t, y, angle):
        g=9.81
        p, v = y
        dpdt = v
        dvdt = g*np.sin(angle)
        return [dpdt, dvdt]

    def integrate_motion(self, p_0 = np.ndarray, v_0=np.ndarray, angle=np.ndarray, t_end=float):
        sol_x = solve_ivp(self.equations_of_motion, [0, t_end], [p_0[0], v_0[0]], args=(angle[0],), t_eval=[t_end])
        sol_y = solve_ivp(self.equations_of_motion, [0, t_end], [p_0[1], v_0[1]], args=(angle[1],), t_eval=[t_end])
    
        return np.array([sol_x.y[0][-1], sol_y.y[0][-1]]), np.array([sol_x.y[1][-1], sol_y.y[1][-1]]) 

    def draw_plate(self):
        pygame.draw.rect(self.screen, self.PLATE_COLOR, (0, 0, self.WIDTH, self.HEIGHT))
        pygame.draw.circle(self.screen, (255,0,0), (int(self.WIDTH/2), int(self.HEIGHT/2)), 50, 1)
        pygame.draw.circle(self.screen, (255,0,0), (int(self.WIDTH/2), int(self.HEIGHT/2)), 100, 1)
    def draw_ball(self, coordinates=np.ndarray):
        pygame.draw.circle(self.screen, self.BALL_COLOR, (int(coordinates[0]), int(coordinates[1])), self.BALL_RADIUS)
    def draw_tilt_line(self, tilt_angle):
        pygame.draw.line(self.screen, (0,0,0), (int(self.WIDTH/2), int(self.HEIGHT/2)), (int(self.WIDTH/2+10*np.rad2deg(tilt_angle[0])), int(self.HEIGHT/2+10*np.rad2deg(tilt_angle[1]))), 3)
        pygame.draw.circle(self.screen, (0,0,0), (int(self.WIDTH/2), int(self.HEIGHT/2)), 4)
