import yaml
from aloha import RosOperator

with open("./config/inference.yaml", "r") as f:
    config = yaml.safe_load(f)

ros_operator = RosOperator(config)

ros_operator.setup_puppet_arm()


