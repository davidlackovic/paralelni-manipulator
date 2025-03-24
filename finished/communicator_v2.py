import serial
import time
import numpy as np
import threading

class SerialCommunication():
    '''Class for serial communication with 3D printer motherboard.
       Parameters:
       - max_speed (default 2500)
       - max_acceleration (default 200)
    '''
    def __init__(self, ser, max_speed=5000, max_acceleration=200, normal_acceleration=50):
        '''Setup sequence:
            -max speed M203
            -max accel M201
            -steps/mm M92
            -set current coords G92'''
        self.ser = ser
        ser.write(bytes('', 'utf-8'))
        time.sleep(7)
        response = ser.readlines()
        print("Printer response:", response)

        ser.write(bytes('G90\n', 'utf-8'))
        time.sleep(0.01)
        ser.write(bytes(f'M203 X{max_speed} Y2{max_speed} Z2{max_speed}\n', 'utf-8'))
        time.sleep(0.01)
        ser.write(bytes(f'M201 X{max_acceleration} Y{max_acceleration} Z{max_acceleration}\n', 'utf-8'))
        time.sleep(0.01)
        ser.write(bytes(f'M204 P{normal_acceleration} T{normal_acceleration}\n', 'utf-8'))
        time.sleep(0.01)
        ser.write(bytes('M92 X100 Y100 Z100\n', 'utf-8'))
        time.sleep(0.01)
        ser.write(bytes('M211 S0\n', 'utf-8')) #disable software endstops
        time.sleep(0.01)
        ser.write(bytes('M121\n', 'utf-8')) #disable physical endstops
        time.sleep(0.01)
        ser.write(bytes('G92 X-6.31111111 Y-6.31111111 Z-6.31111111\n', 'utf-8'))
        time.sleep(0.01)
        ser.write(bytes('G1 X-6.31111111 Y-6.31111111 Z-6.31111111\n', 'utf-8'))
        time.sleep(0.01)
        ser.write(bytes('M115\n', 'utf-8'))
        time.sleep(0.01)
        response = ser.readlines()

        if any("FIRMWARE_NAME" in line.decode() for line in response):
            print(f'Communication with board established. Max speed set to {max_speed}, max acceleration to {max_acceleration}.')
        else:
            print("No valid response received. Check the connection.")



    def move_to_position(self, pos, feedrate=None):
        '''Move axes to a (x, y, z) position.'''
        
        if len(pos) != 3:
            raise(ValueError(f'Wrong input length \"{pos}\". Should be 3-list (x,y,z). x, y or z can be None.'))
        x, y, z = pos
            
        code = 'G1 '
        if x != None:
            code += f'X{x:.5f} '
        if y != None:
            code += f'Y{y:.5f} '
        if z != None:
                code += f'Z{z:.5f} '
        if feedrate != None:
            code += f'F{feedrate:.0f}'
        elif feedrate == None:
            code += 'F700'

        self.ser.write(bytes(f'{code}\n', 'utf-8'))
        print(f'Sent: {code}')
        

    def send_gcode(self, code):
        '''Send simple G-code.'''
        self.ser.write(bytes(f'{code}\n', 'utf-8'))
        print(f'Sent: {code}')
        

    def enable_steppers(self):
        '''Lock steppers in place.'''
        self.ser.write(bytes(f'M17\n', 'utf-8'))
        print(f'Steppers enabled (locked).')



    def disable_steppers(self):
        '''Unlock steppers.'''
        self.ser.write(bytes(f'M18\n', 'utf-8'))
        print(f'Steppers disabled (unlocked).')

    def get_current_position(self):
        '''Prints current position of steppers.'''
        self.ser.write(bytes(f'M114\n', 'utf-8'))
        response = self.ser.readline().decode('utf-8').strip()    
        print("Printer Position:", response)
