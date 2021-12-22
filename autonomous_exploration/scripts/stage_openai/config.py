import argparse

class Config():

    def __init__(self):

        parser = argparse.ArgumentParser(description='RL training for frontier selection')
        parser.add_argument('--num_actions', type=int, default=4, help='num frontiers (fixed)')
        parser.add_argument('--num_states', type=int, default=43, help='num states (fixed)')
        parser.add_argument('--MAX_STEPS', type=int, default=20, help='num steps for episode)')
        parser.add_argument('--NUM_EPISODES', type=int, default=500, help='num episodes')
        parser.add_argument('--TOTAL_EPISODES', type=int, default=500, help='Total episodes')
        parser.add_argument('--maps_gt', default=["map2","map2","map2","map2"], help='vector with name of stage maps for training')
        parser.add_argument('--DISCOUNT_FACTOR', default=0.9, help='vector with name of stage maps for training')
        parser.add_argument('--SOLVED_SCORE', default=2000, help='vector with name of stage maps for training')
        parser.add_argument('--CRITIC_LAMBDA', default=0.9, help='vector with name of stage maps for training')
        self.args = parser.parse_args()

    def parse(self):
        return self.args