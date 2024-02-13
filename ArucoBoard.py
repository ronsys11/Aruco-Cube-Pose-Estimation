import numpy as np
import cv2
import sys
from utils import ARUCO_DICT
import argparse
import time
import pyrealsense2 as rs


aruco_marker_length_meters = 0.037
# creating an aruco board
import numpy as np
import cv2 as cv

# Define object points (modify based on your marker size and orientation)
board_corners = [np.array([[-0.019,0.019,0.027],[0.019,0.019,0.027],[0.019,-0.019,0.027],[-0.019,-0.019,0.027]],dtype=np.float32),
                 np.array([[-0.027,0.019,-0.019],[-0.027,0.019,0.019],[-0.027,-0.019,0.019],[-0.027,-0.019,-0.019]],dtype=np.float32),
                 np.array([[0.027,0.019,0.019],[0.027,0.019,-0.019],[0.027,-0.019,-0.019],[0.027,-0.019,0.019]],dtype=np.float32),
                 np.array([[-0.019,0.027,-0.019],[0.019,0.027,-0.019],[0.019,0.027,0.019],[-0.019,0.027,0.019]],dtype=np.float32)] 
board_ids     = np.array([2 ,0, 1, 6], dtype=np.int32)

board = cv2.aruco.Board( board_corners, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250),board_ids )

# Create Aruco dictionary
dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_100)

def pose_estimation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients, depth_frame):


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters()

    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
 

    
    # If markers are detected
    if len(corners) > 0:
        for i in range(0, len(ids)):
  
            # Estimate pose of each marker and return the values rvec and tvec           
                retval,rvec,tvec = cv2.aruco.estimatePoseBoard( corners, ids, board, k, d,None,None )
                
                
                print(tvec*100)                            
                #R, _ = cv2.Rodrigues(rvec)
                # R is now the 3x3 rotational matrix representing the rotation                
                #print(R)
                # Draw a square around the markers
                cv2.aruco.drawDetectedMarkers(frame, corners)
                # Draw Axis
                cv2.drawFrameAxes(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)
                x_sum = corners[0][0][0][0]+ corners[0][0][1][0]+ corners[0][0][2][0]+ corners[0][0][3][0]
                y_sum = corners[0][0][0][1]+ corners[0][0][1][1]+ corners[0][0][2][1]+ corners[0][0][3][1]        
                x_centerPixel = int(x_sum*.25)
                y_centerPixel = int(y_sum*.25)
                #print(x_centerPixel,y_centerPixel)
                depth_value = depth_frame.get_distance(x_centerPixel,  y_centerPixel)
        
    return frame



if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", "--K_Matrix", required=True, help="Path to calibration matrix (numpy file)")
    ap.add_argument("-d", "--D_Coeff", required=True, help="Path to distortion coefficients (numpy file)")
    ap.add_argument("-t", "--type", type=str, default="DICT_ARUCO_ORIGINAL", help="Type of ArUco tag to detect")
    args = vars(ap.parse_args())

# Ask the user for some input and transmit it

    if ARUCO_DICT.get(args["type"], None) is None:
        print(f"ArUCo tag type '{args['type']}' is not supported")
        sys.exit(0)

    aruco_dict_type = ARUCO_DICT[args["type"]]
    calibration_matrix_path = args["K_Matrix"]
    distortion_coefficients_path = args["D_Coeff"]

    k = np.load(calibration_matrix_path)
    d = np.load(distortion_coefficients_path)

    # Initialize the RealSense camera
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    align_to = rs.stream.color
    align = rs.align(align_to)
    pipeline.start(config)

    while True:
        
        # Wait for frames from the RealSense camera
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame =  aligned_frames.get_color_frame()
        depth_frame =  aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        # Convert RealSense color frame to a format that OpenCV can handle
        frame = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        #hi = client.result
        #if hi== "T-Button Pressed sucka /n":
        output = pose_estimation(frame, aruco_dict_type, k, d, depth_frame)
                
                

            ##output = pose_estimation(frame, aruco_dict_type, k, d, depth_frame)

        cv2.imshow('Estimated Pose', output)
        cv2.imshow('Depth', depth_image)


        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    pipeline.stop()
    cv2.destroyAllWindows()
