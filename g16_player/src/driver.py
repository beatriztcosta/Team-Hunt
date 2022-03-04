#!/usr/bin/env python3

import copy
import math
import rospy
import tf2_ros

from geometry_msgs.msg import Twist
from tf2_geometry_msgs import PoseStamped # **Do not use geometry_msgs. Use this instead for PoseStamped
from sensor_msgs.msg import LaserScan, Image, CameraInfo
from nav_msgs.msg import Odometry
import cv2
import numpy as np
from cv_bridge import CvBridge
import os


if os.name == 'nt':
  import msvcrt
else:
  import tty, termios



COLORS = ['red', 'green', 'blue']
PLAYERS = [[color+str(j) for j in range(1,4)] for color in COLORS] 
RGB = [(255,0,0), (0,255,0), (0,0,255)]




WAFFLE_MAX_LIN_VEL = 5.26
WAFFLE_MAX_ANG_VEL = 1.82
LIN_VEL_STEP_SIZE = 0.1
ANG_VEL_STEP_SIZE = 0.3


WAFFLE_WHEEL_DIST = 0.287








# binaries
CAMERA_B, INPUTS_B, LASER_B = 0b001, 0b010, 0b100 
READY = 0b111


def twistToDiff(speed, angular):
    sl = (2*speed - angular*WAFFLE_WHEEL_DIST)/2
    sr = angular*WAFFLE_WHEEL_DIST + sl
    return [sr, sl]

def diffToTwist(sr, sl):
    return [(sr+sl)/2, (sr-sl)/WAFFLE_WHEEL_DIST]


bridge = CvBridge()
        

class Normalizer():

    def __init__(self, input_size, file_name="normalizer"):
        self.filename=file_name
        self.n = 0
        self.min = np.ones(input_size)*1000
        self.max = np.ones(input_size)*-1000
        try:
            self.loadFromFile()
        except:
            pass
    def observe(self, x):
        self.n += 1
        self.min = np.minimum(self.min, x)
        self.max = np.maximum(self.max, x)

        if self.n%1000==0:
            self.saveToFile()

    def normalize(self, inputs):
        return inputs/np.abs(self.max.clip(min=1e-2))

    def saveToFile(self):
        np.save(self.filename, {
            "min": self.min,
            "max": self.max,
            "n": self.n
        })

    def loadFromFile(self):
        dc = np.load(self.filename+'.npy')
        self.n = dc["n"] 
        self.min = dc["min"] 
        self.max = dc["max"] 

def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a,idx,axis=axis)


class Driver:

    def __init__(self, name, team, normalizer, input_params):

        self.input_params = input_params


        INPUT_SIZE = input_params["RESOLUTION"] + input_params["NUM_SENSORS"] + 2 + 2

        self.goal = PoseStamped()
        self.goal_active = False
        self.team = team

        self.sensors = []
        self.normalizer = normalizer or Normalizer(INPUT_SIZE)
        
        self.speed = np.zeros((2,1))
        
        self.ready = 0b0

        self.inputs = np.zeros((INPUT_SIZE,))

        self.name = name #rospy.get_name()
        print('My player name is ' + self.name)

        self.publisher_command = rospy.Publisher( '/' + self.name + '/cmd_vel', Twist, queue_size=1)

        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)

        self.goal_subscriber = rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goalReceivedCallback)
        self.laser_subscriber = rospy.Subscriber(f"/{name}/scan", LaserScan, self.scanReceivedFallback)
        self.camera_subscriber = rospy.Subscriber(f"/{name}/camera/rgb/image_raw", Image, self.imageReceivedFallback)
        self.odom_subcriber = rospy.Subscriber(f"/{name}/odom", Odometry, self.odomReceivedFallback)
        self.compressed_laser_pub = rospy.Publisher(f"/{name}/compressed_scan", LaserScan, queue_size=10)
        self.camera_info = rospy.wait_for_message(f"/{name}/camera/rgb/camera_info", CameraInfo)

    
    def odomReceivedFallback(self, event):
        twist = event.twist.twist
        sr, sl = twistToDiff(twist.linear.x, twist.angular.z)
        self.speed = np.array([sr,sl])
        self.inputs[-4:] =  np.array([sr, sl, sr+sl, sr-sl])
        self.ready |= INPUTS_B
        self.act()

    def imageReceivedFallback(self, image):
       
        RESOLUTION = self.input_params["RESOLUTION"]
        SAMPLES = 640/RESOLUTION
        HALF_SAMPLES = SAMPLES/2

        image = bridge.imgmsg_to_cv2(image, desired_encoding='bgr8')
        teamtrackers = []
        kernel = np.ones((2,2))
        for color_idx in range(3):
            values = [0]*RESOLUTION
            lower = np.array([0,0,0]) 
            upper = np.array([10,10,10])
            lower[2-color_idx] = 90
            upper[2-color_idx] = 120
            mask = cv2.threshold(cv2.inRange(image, lower, upper), 128, 255, cv2.THRESH_BINARY)[1]
            mask = cv2.erode(mask, kernel)
            mask = cv2.dilate(mask, kernel)
            mask = mask.sum(axis=0)/255
            mask[mask<2] = 0
            for i in range(640):
                idxsmall = int((i-HALF_SAMPLES)/SAMPLES)
                p = ((i-HALF_SAMPLES)%SAMPLES)/SAMPLES
                if idxsmall>=0:
                    modifier = 0.1+0.9*p
                    values[idxsmall] += modifier*mask[i]
                idxbig = idxsmall+1
                if idxbig<RESOLUTION:
                    modifier = 0.1+0.9*(1-p)
                    values[idxbig] += modifier*mask[i]
            values = np.array(values)/100
            teamtrackers.append(values)
        
        self.inputs[:RESOLUTION] = teamtrackers[(self.team+1)%3] - teamtrackers[(self.team-1)%3]
        self.ready |= CAMERA_B  
        self.act()

        
    
    def scanReceivedFallback(self, msg):


        NUM_SENSORS = self.input_params["NUM_SENSORS"]
        RANGE_MAX = self.input_params["RANGE_MAX"]
        RANGE_MIN = self.input_params["RANGE_MIN"]

        sensors = [math.inf]*NUM_SENSORS
        for idx, dist in enumerate(msg.ranges):

            if dist < 0.1 or dist==math.inf:
                continue

            theta = msg.angle_min + msg.angle_increment * idx

            if theta>math.pi:
                theta = min(theta - math.pi*2, 0) 
            
            if theta>=RANGE_MIN and theta<=RANGE_MAX:
                idx = int( NUM_SENSORS*(theta-RANGE_MIN)/(RANGE_MAX-RANGE_MIN) )

                sensors[idx] = min(dist, sensors[idx])

        # publish
        jumps = (RANGE_MAX-RANGE_MIN)/(NUM_SENSORS-1)
        laserscan = LaserScan()
        laserscan.header = msg.header
        laserscan.angle_min = RANGE_MIN + jumps/2
        laserscan.angle_max = RANGE_MAX - jumps/2
        laserscan.angle_increment = jumps
        laserscan.ranges = sensors
        laserscan.range_min = min(sensors)
        laserscan.range_max = max([s for s in sensors if s<math.inf]or[0])
        laserscan.intensities = [0]*NUM_SENSORS        
        self.compressed_laser_pub.publish(laserscan)

        sensors = np.array(sensors).clip(max=100)
        sensors = 10/(sensors+1) 
        sensors[sensors<=.2] = 0
        self.sensors = sensors

        self.ready |= LASER_B
        self.act()
        


    def goalReceivedCallback(self, msg):
        print('Received new goal on frame id' + msg.header.frame_id)
        target_frame = self.name + '/odom'
        try:

            self.goal = self.tf_buffer.transform(msg, target_frame, rospy.Duration(1))
            self.goal_active = True
            rospy.logwarn('Setting new goal')
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            self.goal_active = False
            rospy.logerr('Could not transform goal from ' + msg.header.frame_id + ' to ' + target_frame + '. Will ignore this goal.')


    def act(self):

        if self.ready != READY:
            return

