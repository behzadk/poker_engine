import numpy as np
from agent import Agent
from player import Player
from table import Table

class DicePlayer(object):
    """docstring for DicePlayer"""
    def __init__(self, player_id, starting_stack):

        training=False
        epsilon_testing=0.00
        checkpoint_dir="./checkpoint_bvb_0"
        
        self.action_type_space = ['FOLD', 'ALL_IN']

        self.stack = starting_stack
        self.prev_stack = starting_stack
        self.agent = Agent(agent_name=player_id, checkpoint_dir=checkpoint_dir, epsilon_testing=epsilon_testing,
            action_names=self.action_type_space, training=training, state_shape=[1,],
            render=False, use_logging=True)


    def get_action(self, game_state):
        action_idx = self.agent.get_action(game_state, self, self)
        return action_idx


##
# 
# Dice is rolled. Player money doubled if it is a 6
#
##
class GameEnv:
    def __init__(self, players):
        self.players = players
        self.dice_value = None

        self.games_played = 0
        self.p_0_best = 0


    def play_game(self):
        p_0 = self.players[0]
        self.dice_value = np.random.randint(0, 6)

        game_state = np.array(self.dice_value, dtype=np.float32).reshape(1, -1)

        action_idx = p_0.get_action(game_state)

        # print(action_idx)

        # No bet
        if self.dice_value == 0 and action_idx == 0:
            game_reward = p_0.prev_stack * 2
            self.p_0_best += 1

        elif self.dice_value != 0 and action_idx == 1:
            game_reward = p_0.prev_stack
            self.p_0_best += 1

        else:
            game_reward =0

        self.games_played += 1

        # print(self.games_played)
        print(self.p_0_best/self.games_played)
        print("")

        p_0.agent.update_replay_memory(end_hand_reward=game_reward)


    def reset_players(self, stack):
        for p in self.players:
            p.starting_stack = stack
            p.prev_stack = stack


def main():
    p_0 = DicePlayer('p_0', 5)

    players = [p_0]
    game_env = GameEnv(players)

    for i in range(1, int(1e12)):
        game_env.play_game()
        game_env.reset_players(5)




if __name__ == "__main__":
    main()


