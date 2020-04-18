import numpy as np
from agent import Agent
from encoding_functions import StateEncoder
from deuces import Card

## Action format:
# ['ACTION TYPE', int(value to transfer to pot)]
# e.g ['RAISE', 10], ['FOLD', 0]
# 
def random_min_raise(table, this_player, round_actions):
    current_bet = table.current_bet
    min_raise = table.big_blind


    action_type_space = ['CALL', 'RAISE', 'CHECK', 'FOLD']

    # No bets made, can only check or raise
    if current_bet == 0:
        action_type_space = ['RAISE', 'CHECK']

    elif current_bet > this_player.stack:
        action_type_space = ['CALL', 'FOLD']


    else:
        action_type_space = ['CALL', 'RAISE', 'FOLD']

    chosen_action_type = np.random.choice(action_type_space)

    action_value = 0

    # If call, value to transfer is current bet
    if chosen_action_type == 'CALL':
        if current_bet > this_player.stack:
            action_value = this_player.stack

        else:
            action_value = current_bet

    # If raise, apply min raise
    elif chosen_action_type == 'RAISE':
        if min_raise + current_bet > this_player.stack:
            min_raise = this_player.stack

        action_value = current_bet + min_raise

    # If fold, value to transfer is 0
    elif chosen_action_type == 'FOLD':
        action_value = 0

    # If check, value to transfer is 0
    elif chosen_action_type == 'CHECK':
        action_value = 0

    else:
        print("Invalid action type, quitting... ")
        exit()

    ret_action = [chosen_action_type, action_value]

    return ret_action


def always_raise(table, this_player, round_actions):
    current_bet = table.current_bet
    min_raise = table.big_blind

    max_raise = (this_player.stack - current_bet) * 0.25

    action_type_space = ['CALL', 'RAISE', 'CHECK', 'FOLD']

    if this_player.stack < max_raise:
        chosen_action_type = 'CALL'
        action_value = current_bet

    else:
        chosen_action_type = 'RAISE'
        action_value = current_bet + np.random.uniform(min_raise, max_raise)

    ret_action = [chosen_action_type, action_value]

    return ret_action


def huamn_action(table, this_player, round_actions):
    # this_player.display_game_state(table, round_actions)
    current_bet = table.current_bet

    action_type_space = ['CALL', 'RAISE', 'CHECK', 'FOLD']

    # No bets made, can only check or raise
    if current_bet == 0:
        action_type_space = ['RAISE', 'CHECK']

    elif current_bet > this_player.stack:
        action_type_space = ['CALL', 'FOLD']

    else:
        action_type_space = ['CALL', 'RAISE', 'FOLD']

    while True:

        # check if d1a is equal to one of the strings, specified in the list
        try:
            action_type_idx = int(input("Choose action index: "))
            chosen_action_type = action_type_space[action_type_idx]
            break

        except (IndexError, ValueError):
            print("invalid index, please choose again")

    # If call, value to transfer is current bet
    if chosen_action_type == 'CALL':
        if current_bet > this_player.stack:
            action_value = this_player.stack

        else:
            action_value = current_bet

    # If raise, apply min raise
    elif chosen_action_type == 'RAISE':
        while True:

            # check if d1a is equal to one of the strings, specified in the list
            try:
                raise_value = int(input("Choose raise value: "))
                chosen_action_type = action_type_space[action_type_idx]

            except (IndexError, ValueError):
                print("invalid index, please choose again")
                continue

            if raise_value + current_bet > this_player.stack:
                print("Going all in")
                raise_value = this_player.stack

            break

        action_value = current_bet + raise_value

    # If fold, value to transfer is 0
    elif chosen_action_type == 'FOLD':
        action_value = 0

    # If check, value to transfer is 0
    elif chosen_action_type == 'CHECK':
        action_value = 0

    else:
        print("Invalid action type, quitting... ")
        exit()

    ret_action = [chosen_action_type, action_value]

    return ret_action


class RlBot:
    def __init__(self, agent_name, training, epsilon_testing, checkpoint_dir):
        self.action_type_space = ['CALL', 'ALL_IN', 'CHECK', 
        'FOLD', 'RAISE_1', 'RAISE_2', 'RAISE_4', 'RAISE_8']
        
        self.action_type_space = ['CALL', 'ALL_IN', 'CHECK', 
        'FOLD', 'RAISE_1', 'RAISE_1_5', 'RAISE_2', 'RAISE_2_5', 
        'RAISE_3', 'RAISE_3_5', 'RAISE_4', 'RAISE_4_5']

        self.start_episode_stack = 0
        self.state_encoder = StateEncoder(checkpoint_dir)
        state_shape = self.state_encoder.get_state_shape()

        self.agent = Agent(agent_name=agent_name, checkpoint_dir=checkpoint_dir, epsilon_testing=epsilon_testing,
            action_names=self.action_type_space, training=training, state_shape=[state_shape,],
            render=False, use_logging=True)

    
    def start_episode(self, init_stack):
        self.start_episode_stack = init_stack

    def get_action(self, table, this_player, round_actions):
        current_bet = table.current_bet
        min_raise = table.big_blind
        max_raise = this_player.stack - current_bet


        state = self.state_encoder.encode_state(this_player, table)

        # Get action from agent
        action_idx = self.agent.get_action(this_player, table, state)

        chosen_action_type = self.action_type_space[action_idx]

        # if self.agent.agent_name == "bvb_7":
        #     Card.print_pretty_cards(this_player.hand)
        #     Card.print_pretty_cards(table.board)
        #     print(chosen_action_type)
        #     print("")


        # print(self.agent.agent_name)
        # Card.print_pretty_cards(this_player.hand)
        # Card.print_pretty_cards(table.board)
        # print(chosen_action_type)
        # print("")


        if chosen_action_type == "CALL":
            action_value = current_bet

        elif chosen_action_type == "ALL_IN":
            action_value = this_player.stack

            if action_value > current_bet:
                chosen_action_type = "RAISE"

        elif chosen_action_type == "CHECK" or chosen_action_type == "FOLD":
            action_value = 0

        elif chosen_action_type == "RAISE_1":
            action_value = current_bet + table.big_blind
            chosen_action_type = "RAISE"

        elif chosen_action_type == "RAISE_1_5":
            action_value = current_bet + table.big_blind * 1.5
            chosen_action_type = "RAISE"

        elif chosen_action_type == "RAISE_2":
            action_value = current_bet + table.big_blind * 2
            chosen_action_type = "RAISE"

        elif chosen_action_type == "RAISE_2_5":
            action_value = current_bet + table.big_blind * 2.5
            chosen_action_type = "RAISE"

        elif chosen_action_type == "RAISE_3":
            action_value = current_bet + table.big_blind * 3
            chosen_action_type = "RAISE"

        elif chosen_action_type == "RAISE_3_5":
            action_value = current_bet + table.big_blind * 3.5
            chosen_action_type = "RAISE"

        elif chosen_action_type == "RAISE_4":
            action_value = current_bet + table.big_blind * 4
            chosen_action_type = "RAISE"

        elif chosen_action_type == "RAISE_4_5":
            action_value = current_bet + table.big_blind * 4.5
            chosen_action_type = "RAISE"

        else:
            print("invalid action, exiting...")
            exit()

        ret_action = [chosen_action_type, action_value]
        return ret_action, state, fold_state






