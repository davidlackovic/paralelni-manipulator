import serial
import time
import numpy as np
from finished.funkcije_v2 import izracun_kotov

serial_port = 'COM10'

#Dejanski parametri
b = 0.071014 # m
p = 0.223723 # m
l_1 = 0.2 # m
l_2 = 0.23481 # m

ser = serial.Serial(serial_port, 115200, timeout=1)

ser.write(bytes("", 'utf-8'))
time.sleep(2)

ser.write(bytes('G90\n', 'utf-8'))
time.sleep(0.01)
ser.write(bytes('M92 X100 Y100 Z100\n', 'utf-8'))
time.sleep(0.01)
ser.write(bytes('G92 X0 Y0 Z0\n', 'utf-8'))
time.sleep(0.01)
ser.write(bytes('G1 X0 Y0 Z0\n', 'utf-8'))
time.sleep(0.01)

X = Y = h = None

while True:
    line = input("Format: X[v stopinjah] Y[v stopinjah] Z[v metrih]:   ")
    if line.startswith('S'):
        break
    parts = line.split()

    for part in parts:
        if part.startswith('X'):
            X = float(part[1:])
        elif part.startswith('Y'):
            Y = float(part[1:])
        elif part.startswith('Z'):
            h = float(part[1:])

    psi_x = np.deg2rad(float(X))
    psi_y = np.deg2rad(float(Y))

    angles = list(izracun_kotov(b, p, l_1, l_2, h, psi_x, psi_y))

    for i in range(0,3):
        angles[i] = -angles[i]*4/1.8*16/100
    
    output = f'G0 X{angles[0]} Y{angles[1]} Z{angles[2]} F700\n'
    print(output)
    ser.write(bytes(output, 'utf-8'))

ser.close()
    





