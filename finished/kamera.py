import cv2
import numpy as np
    
class CV2Wrapper():
    def __init__(self, camera_index=0, window_name='Webcam feed', camera_matrix=np.array([0]), distortion_coefficients=np.array([0])):
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        self.window_name=window_name
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) 
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.smooth_x = None
        self.smooth_y = None
        self.smooth_r = None
        self.cx_i = 0.0
        self.cx_i1 = 0.0
        self.cx_i2 = 0.0
        self.cy_i = 0.0
        self.cy_i1 = 0.0
        self.cy_i2 = 0.0

        self.old_cx = 0
        self.old_cy = 0

        self.camera_matrix = camera_matrix
        self.distortion_coefficients = distortion_coefficients

        self.target_pos = np.array([0,0])

        # offsets for setting reference frame
        self.x_offset = 332
        self.y_offset = 282


        if not self.cap.isOpened():
            print('Cap failed to open')
            exit()

        
    def create_window(self):
        cv2.namedWindow(self.window_name)

    def adjust_exposure(self, value):
        self.exposure_value = 30*value / 200 - 15
        self.cap.set(cv2.CAP_PROP_EXPOSURE, self.exposure_value)

    def exposure_calibration(self,  lower_color=np.array([0,0,0]), upper_color=np.array([0,0,0])):
        '''Opens a window with a slider to calibrate exposure. If lower_color and upper_color are passed tracking of the ball will be tested too.'''
        cv2.namedWindow("Exposure calibration")
        cv2.createTrackbar("Exposure", "Exposure calibration", 100, 200, self.adjust_exposure)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, -8.1)

        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            processed_frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)
            processed_frame = cv2.undistort(processed_frame, self.camera_matrix, self.distortion_coefficients)
            hsv = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.medianBlur(gray, 5)

            if np.any(lower_color != 0) and np.any(upper_color != 0):
                circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=0.8, minDist=30, param1=40, param2=40, minRadius=10, maxRadius=30)
                if circles is not None:
                    circles = np.uint16(np.around(circles))
                    for (x, y, r) in circles[0, :]:
                        center_color = np.mean(hsv[y-5:y+5, x-5:x+5], axis=(0,1))
                        if (lower_color[0] <= center_color[0] <= upper_color[0] and
                                lower_color[1] <= center_color[1] <= upper_color[1] and
                                lower_color[2] <= center_color[2]<= upper_color[2]):
                            center_x, center_y, radius = x, y, r
                            cv2.circle(processed_frame, (int(center_x), int(center_y)), int(radius), (0, 255, 0), 1)

            cv2.imshow("Exposure calibration", processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print(f'Exposure calibration complete, exposure set to {self.exposure_value}')
                break
        cv2.destroyWindow('Exposure calibration')

    def smooth_function(self, new_value, smooth_value, alpha):
        """Function to smooth the bounding box values using exponential moving average."""
        if smooth_value is None:
            return new_value
        return alpha * new_value + (1 - alpha) * smooth_value


    def process_frame(self, frame, lower_color, upper_color, alpha, flickering=3):
        '''Process the frame and return processed frame, ball location and velocity.
        '''
        vx, vy = 0, 0
        resized_frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)
        processed_frame = cv2.undistort(resized_frame, self.camera_matrix, self.distortion_coefficients)
        hsv = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, 5)

        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=0.8, minDist=30, param1=40, param2=40, minRadius=10, maxRadius=30)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for (x, y, r) in circles[0, :]:
                center_color = hsv[y,x]
                if (lower_color[0] <= center_color[0] <= upper_color[0] and
                        lower_color[1] <= center_color[1] <= upper_color[1] and
                        lower_color[2] <= center_color[2]<= upper_color[2]):

                    center_x,center_y, radius = x, y, r                   
                    self.smooth_x = self.smooth_function(center_x, self.smooth_x, alpha)
                    self.smooth_y = self.smooth_function(center_y, self.smooth_y, alpha)
                    self.smooth_r = self.smooth_function(radius, self.smooth_r, alpha)
                
                    # calculate ball position
                    cx, cy = self.smooth_x, self.smooth_y
                    offset_cx = np.float64(cx) - np.float64(self.x_offset)
                    offset_cy = np.float64(cy) - np.float64(self.y_offset)
                    self.cx_i, self.cx_i1, self.cx_i2 = offset_cx, self.cx_i, self.cx_i1
                    self.cy_i, self.cy_i1, self.cy_i2 = offset_cy, self.cy_i, self.cy_i1

                    cx_arr = np.array([self.cx_i, self.cx_i1, self.cx_i2])
                    cy_arr = np.array([self.cy_i, self.cy_i1, self.cy_i2])

                    # calculate velocity vector
                    vx = int(np.gradient(cx_arr)[0])
                    vy = int(np.gradient(cy_arr)[0])

                    cv2.circle(processed_frame, (int(cx), int(cy)), 2, (0, 0, 0), 2)
                    cv2.circle(processed_frame, (int(cx), int(cy)), int(self.smooth_r), (0, 255, 0), 1)
                    cv2.circle(processed_frame, (self.target_pos[0]+self.x_offset, self.target_pos[1]+self.y_offset), 3, (0,0,255), -1)



                    return processed_frame, np.array([int(offset_cx), int(offset_cy)]), np.array([vx, vy])

            else:
                print("No ball detected")
                return processed_frame, np.array([None, None]), np.array([None, None])
        else:
                print("No contour detected")
                return processed_frame, np.array([None, None]), np.array([None, None])

    def mouse_callback(self, event, x, y, flags, param):
        '''When left click is detected changes target_position to cursor coordinates.
        
        '''
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f'Target position set to {int(x- self.x_offset), int(y- self.y_offset)}')
            self.target_pos = np.array([int(x- self.x_offset), int(y- self.y_offset)])

    def set_mouse_callback(self):    
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

   
        