from collections import deque
import math
import threading
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState
from std_msgs.msg import Header

class RosOperator:
    def __init__(self, config):
        self.config = config

        self.bridge = CvBridge()
        self.img_front_deque = deque()
        self.img_right_deque = deque()
        self.img_left_deque = deque()

        self.puppet_arm_right_deque = deque()
        self.puppet_arm_left_deque = deque()
        
        self.img_deque_list = [self.img_front_deque, self.img_right_deque, self.img_left_deque]
        self.puppet_arm_deque_list = [self.puppet_arm_left_deque, self.puppet_arm_right_deque]
        self.deque_list = self.img_deque_list + self.puppet_arm_deque_list

        rospy.Subscriber(self.config["img_left_topic"], Image, self.img_left_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.config["img_right_topic"], Image, self.img_right_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.config["img_front_topic"], Image, self.img_front_callback, queue_size=1000, tcp_nodelay=True)

        rospy.Subscriber(self.config["puppet_arm_left_topic"], JointState, self.puppet_arm_left_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.config["puppet_arm_right_topic"], JointState, self.puppet_arm_right_callback, queue_size=1000, tcp_nodelay=True)
        
        rospy.init_node("joint_state_publisher", anonymous=True)

        self.puppet_arm_left_publisher = rospy.Publisher(self.config["puppet_arm_left_topic"], JointState, queue_size=10)
        self.puppet_arm_right_publisher = rospy.Publisher(self.config["puppet_arm_right_topic"], JointState, queue_size=10)

        self.puppet_arm_publish_thread = None
        self.puppet_arm_publish_lock = threading.Lock()
        self.puppet_arm_publish_lock.acquire()

    def setup_puppet_arm(self):
        left0 = [-0.00133514404296875, 0.00209808349609375, 0.01583099365234375, -0.032616615295410156, -0.00286102294921875, 0.00095367431640625, 3.557830810546875]
        right0 = [-0.00133514404296875, 0.00438690185546875, 0.034523963928222656, -0.053597450256347656, -0.00476837158203125, -0.00209808349609375, 3.557830810546875]
        left1 = [-0.00133514404296875, 0.00209808349609375, 0.01583099365234375, -0.032616615295410156, -0.00286102294921875, 0.00095367431640625, -0.3393220901489258]
        right1 = [-0.00133514404296875, 0.00247955322265625, 0.01583099365234375, -0.032616615295410156, -0.00286102294921875, 0.00095367431640625, -0.3397035598754883]

        self.puppet_arm_publish_continuous(left0, right0)
        input("Enter any key to continue :")
        self.puppet_arm_publish_continuous(left1, right1)

    def puppet_arm_publish(self, left_target, right_target):
        joint_state_msg = JointState()
        joint_state_msg.header = Header()
        joint_state_msg.header.stamp = rospy.Time.now()
        joint_state_msg.name = ["joint0", "joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        joint_state_msg.position = left_target
        self.puppet_arm_left_publisher.publish(joint_state_msg)
        joint_state_msg.position = right_target
        self.puppet_arm_right_publisher.publish(joint_state_msg)

    def puppet_arm_publish_continuous(self, left_target, right_target):
        rate = rospy.Rate(self.config["publish_rate"])

        state_dimension = len(left_target)

        current_left = None
        current_right = None
        
        while not rospy.is_shutdown():
            if len(self.puppet_arm_left_deque) != 0:
                current_left = list(self.puppet_arm_left_deque[-1].position)
            
            if len(self.puppet_arm_right_deque) != 0:
                current_right = list(self.puppet_arm_right_deque[-1].position)
            
            if current_left is None or current_right is None:
                rate.sleep()
                continue
            else:
                break
        
        step_num = 0
        for i in range(state_dimension):
            left_require_step_num = math.floor(abs(left_target[i] - current_left[i]) / self.config["arm_steps_length"][i])
            right_require_step_num = math.floor(abs(right_target[i] - current_right[i]) / self.config["arm_steps_length"][i])

            if left_require_step_num > step_num:
                step_num = left_require_step_num
            if right_require_step_num > step_num:
                step_num = right_require_step_num

        left_via_list = np.linspace(current_left, left_target, step_num)
        right_via_list = np.linspace(current_right, right_target, step_num)

        for i in range(step_num):
            if rospy.is_shutdown():
                return
            
            left_via_point = left_via_list[i]
            right_via_point = right_via_list[i]

            joint_state_msg = JointState()
            joint_state_msg.header = Header()
            joint_state_msg.header.stamp = rospy.Time.now()
            joint_state_msg.name = ["joint0", "joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
            joint_state_msg.position = left_via_point
            self.puppet_arm_left_publisher.publish(joint_state_msg)
            joint_state_msg.position = right_via_point
            self.puppet_arm_right_publisher.publish(joint_state_msg)

            rate.sleep()


    def puppet_arm_publish_continuous_thread(self, left, right):
        if self.puppet_arm_publish_thread is not None:
            self.puppet_arm_publish_lock.release()
            self.puppet_arm_publish_thread.join()
            self.puppet_arm_publish_lock.acquire(False)
            self.puppet_arm_publish_thread = None
        self.puppet_arm_publish_thread = threading.Thread(target=self.puppet_arm_publish_continuous, args=(left, right))
        self.puppet_arm_publish_thread.start()

    def get_frame(self):
        # 队列不为空
        for deque in self.deque_list:
            if len(deque) == 0:
                return False

        # 获取最新时间戳
        frame_time = min([deque[0].header.stamp.to_sec() for deque in self.img_deque_list])

        # 确认该时间戳可以获取到数据
        for deque in self.deque_list:
            if deque[0].header.stamp.to_sec() > frame_time:
                return False

        # 去掉过期数据
        for deque in self.deque_list:
            while len(deque) > 0 and deque[0].header.stamp.to_sec() < frame_time:
                deque.popleft()
            
            if len(deque) == 0:
                return False

        # 获取该时间戳的数据
        frame = []        
        for deque in self.img_deque_list:
            frame.append(self.bridge.imgmsg_to_cv2(deque.popleft(), "passthrough"))
        
        for deque in self.puppet_arm_deque_list:
            frame.append(deque.popleft())
        
        return frame

    def img_left_callback(self, msg):
        if len(self.img_left_deque) >= 2000:
            self.img_left_deque.popleft()
        self.img_left_deque.append(msg)

    def img_right_callback(self, msg):
        if len(self.img_right_deque) >= 2000:
            self.img_right_deque.popleft()
        self.img_right_deque.append(msg)

    def img_front_callback(self, msg):
        if len(self.img_front_deque) >= 2000:
            self.img_front_deque.popleft()
        self.img_front_deque.append(msg)

    def puppet_arm_left_callback(self, msg):
        if len(self.puppet_arm_left_deque) >= 2000:
            self.puppet_arm_left_deque.popleft()
        self.puppet_arm_left_deque.append(msg)

    def puppet_arm_right_callback(self, msg):
        if len(self.puppet_arm_right_deque) >= 2000:
            self.puppet_arm_right_deque.popleft()
        self.puppet_arm_right_deque.append(msg)
