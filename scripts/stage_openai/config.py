import argparse

class Config():

    def __init__(self):

        parser = argparse.ArgumentParser(description='RL training for frontier selection')
        parser.add_argument('--num_actions', type=float, default=1.0, help='num frontiers (fixed)')
        parser.add_argument('--max_action_count', type=int, default=4, help='num actions for epoch (how many times check all frontiers)')
        parser.add_argument('--n_init', type=int, default=1, help='num start positions before change the map')
        parser.add_argument('--maps_gt', default=["map0","map1"], help='vector with name of stage maps for training')
        self.args = parser.parse_args()

    def parse(self):
        return self.args