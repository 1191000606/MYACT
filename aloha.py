from collections import deque
import math
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

        # rospy.Subscriber表示订阅该信道，可以从该信道获取信息。roslaunch astra_camera multi_camera.launch之后，这三个信道都会有图像信息
        rospy.Subscriber(self.config["img_left_topic"], Image, self.img_left_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.config["img_right_topic"], Image, self.img_right_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.config["img_front_topic"], Image, self.img_front_callback, queue_size=1000, tcp_nodelay=True)

        # 这里表示定义从臂信息。roslaunch arm_control puppet.launch之后，这两个信道都会有信息
        rospy.Subscriber(self.config["puppet_arm_left_topic"], JointState, self.puppet_arm_left_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.config["puppet_arm_right_topic"], JointState, self.puppet_arm_right_callback, queue_size=1000, tcp_nodelay=True)
        
        rospy.init_node("joint_state_publisher", anonymous=True)

        # 这里表示发布信息，从臂会从/master/joint_left，/mater/joint_right信道获取信息，然后基于获取的信息控制自身。这种情况下不能再roslaunch arm_control master.launch了，不然会有会有两个地方向/master信道发消息，一个是来自与遥操作的物理主臂的变化，一个是这里的ACT模型的推理，这样会导致冲突。
        self.puppet_arm_left_publisher = rospy.Publisher(self.config["puppet_arm_left_cmd_topic"], JointState, queue_size=10)
        self.puppet_arm_right_publisher = rospy.Publisher(self.config["puppet_arm_right_cmd_topic"], JointState, queue_size=10)

        # 不过的话，如果研究reply_action的代码，就会发现，reply需要roslaunch arm_control puppet.launch，但是不能launch master.launch。然后在reply_action.py代码中，会有puppet_arm_publisher，和master_arm_publisher。这样的话只有一个地方向master信道发消息，就是reply_action.py中的master_arm_publisher。至于从臂，从臂会从/master/joint信道获取信息，然后基于获取到的信息控制自身。然后从臂自身位置的变化也会发布到/puppet_arm信道上。reply_action.py代码中又一个puppet_arm_publisher发消息，不会因为冲突导致程序结束，不过至少会导致puppet_arm信道上消息频率翻倍。不过reply的过程中不会从从臂信道采集信息，所以就还好

        # 这里的act程序是会从从臂信道采集数据的。

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
        for i in range(len(left_target)):
            left_require_step_num = math.floor(abs(left_target[i] - current_left[i]) / self.config["arm_steps_length"][i])
            right_require_step_num = math.floor(abs(right_target[i] - current_right[i]) / self.config["arm_steps_length"][i])

            if left_require_step_num > step_num:
                step_num = left_require_step_num
            if right_require_step_num > step_num:
                step_num = right_require_step_num

        left_via_list = np.linspace(current_left, left_target, step_num)
        right_via_list = np.linspace(current_right, right_target, step_num)

        print("left_move: ", np.array(current_left) - np.array(left_target))
        print("right_move: ", np.array(current_right) - np.array(right_target))

        for i in range(step_num):
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

    def get_frame(self):
        # 队列不为空
        for deque in self.deque_list:
            if len(deque) == 0:
                return None

        # 获取最新时间戳
        frame_time = min([deque[-1].header.stamp.to_sec() for deque in self.deque_list])

        # 确认该时间戳可以获取到数据
        for deque in self.deque_list:
            if deque[0].header.stamp.to_sec() > frame_time:
                return None

        # 去掉过期数据
        for deque in self.deque_list:
            while len(deque) > 0 and deque[0].header.stamp.to_sec() < frame_time:
                deque.popleft()
            
            if len(deque) == 0:
                return None

        # 获取该时间戳的数据
        frame = []        
        for deque in self.img_deque_list:
            frame.append(self.bridge.imgmsg_to_cv2(deque.popleft(), "passthrough"))
        
        for deque in self.puppet_arm_deque_list:
            frame.append(deque.popleft().position)
        
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
