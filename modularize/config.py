import os
import yaml

with open(f'{os.path.dirname(os.path.abspath(__file__))}/config.yml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
