import numpy as np
import pygame
import sys
import matplotlib.pyplot as plt

class PygameDraw():
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Simulacija")
        clock = pygame.time.Clock() 
    
    def draw_plate():
        pygame.draw.rect(self.screen, PLATE_COLOR, (0, 0, WIDTH, HEIGHT))
        pygame.draw.circle(self.screen, (255,0,0), (int(WIDTH/2), int(HEIGHT/2)), 50, 1)
        pygame.draw.circle(self.screen, (255,0,0), (int(WIDTH/2), int(HEIGHT/2)), 100, 1)


    def draw_ball(coordinates=np.ndarray):
        pygame.draw.circle(self.screen, BALL_COLOR, (int(coordinates[0]), int(coordinates[1])), BALL_RADIUS)

    def draw_tilt_line(tilt_angle):
        pygame.draw.line(self.screen, (0,0,0), (int(WIDTH/2), int(HEIGHT/2)), (int(WIDTH/2+10*np.rad2deg(tilt_angle[0])), int(HEIGHT/2+10*np.rad2deg(tilt_angle[1]))), 3)
        pygame.draw.circle(self.screen, (0,0,0), (int(WIDTH/2), int(HEIGHT/2)), 4)
