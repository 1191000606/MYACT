import yaml
import rospy

def main(config):
    rospy.init_node("inference_node")
    


if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    main(config)
