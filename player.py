import numpy as np
import pandas as pd
from deuces import Card
from deuces import Deck
from deuces import Evaluator
import os

class Player:
    def __init__(self, player_id, starting_stack, policy):
        self.id = player_id
        self.hand = None
        self.hand_actions = []
        self.active = True

        self.stack = starting_stack
        self.prev_stack = starting_stack
        self.hand_rank = None
        self.contribution_to_pot = 0

        self.all_in = False

        # Playing position is relative to dealer chip. 
        # dealer chip is index [-1]
        self.playing_position = None
        self.policy = policy

        self.stage_actions = []
        self.hand_actions = []

        self.actions_df = self.init_actions_dataframe()
        self.output_path = "./output/player_actions/player_" + str(self.id) + "_actions.csv"



    def init_actions_dataframe(self):
        columns = [
        "hand_idx", "pos_idx", "hand_num_1", "hand_suit_1", 
        "hand_num_2", "hand_suit_2", "board_num_1", "board_suit_1", 
        "board_num_2", "board_suit_2", "board_num_3", "board_suit_3", 
        "board_num_4", "board_suit_4", "board_num_5", "board_suit_5", 
        "stage", "current_stack", "pot", "action_type", "action_value", 
        "current_bet", "num_active_players", "net_stack"
        ]

        df = pd.DataFrame(columns=columns)

        return df


    ##
    # evaluates player hand
    ##
    def evaluate_hand(self, board, eval_obj):
        self.hand_rank = eval_obj.evaluate(board, self.hand)


    def generate_action_data(self, table):

        if self.stage_actions[-1][0] == "ALL_IN":
            return 0

        hand_idx = table.hand_idx
        pos_idx = self.playing_position
        hero_cards = [ [Card.int_to_str(x)[0], Card.int_to_str(x)[1]] for x in self.hand]
        board =  [ [Card.int_to_str(x)[0], Card.int_to_str(x)[1]] for x in table.board]

        # Padding board
        board += ["__"] * (5 - len(board))
        
        stage = table.stage
        stack = self.stack

        pot = table.current_pot
        action_type = self.stage_actions[-1][0]
        action_value = self.stage_actions[-1][1]
        current_bet = table.current_bet
        num_active_players = len(table.get_active_players())
        net_stack = np.nan

        flatten = lambda l: [item for sublist in l for item in sublist]
        
        data_list = []
        data_list.append(hand_idx)
        data_list.append(pos_idx)
        data_list += flatten(hero_cards)
        data_list += flatten(board)
        data_list.append(stage)
        data_list.append(stack)
        data_list.append(pot)
        data_list.append(action_type)
        data_list.append(action_value)
        data_list.append(current_bet)
        data_list.append(num_active_players)
        data_list.append(net_stack)

        # self.actions_df.append(data_list)
        self.actions_df.loc[len(self.actions_df), :] = data_list


    def update_action_data_net_stack(self, hand_idx):
        net_stack = self.stack - self.prev_stack
        self.actions_df.loc[self.actions_df['hand_idx'] == hand_idx, "net_stack"] = net_stack

    ##
    # Returns an action committed by a player.
    # actions are chosen from an action_policy script.
    ##
    def get_action(self, table, table_state, round_actions):
        curent_bet = table.current_bet

        chosen_action = [self.id]

        if self.stack <= 0:
            self.all_in = True
            chosen_action += ['ALL_IN', 0]

        else:
            chosen_action += self.policy(table_state, self, round_actions)
            
            self.stack -= chosen_action[2]

            print(chosen_action)
            if chosen_action[1] == 'FOLD':
                self.active = False

        self.stage_actions.append(chosen_action)
        self.generate_action_data(table)
        self.contribution_to_pot += chosen_action[2]

        return chosen_action


    def display_game_state(self, table, round_actions):
        evaluator = Evaluator()

        if len(table.board) > 0:
            p_score = evaluator.evaluate(table.board, self.hand)
            p_class = evaluator.get_rank_class(p_score)
            p_string = evaluator.class_to_string(p_class)

        else:
            p_string = ""

        os.system('clear')
        print(round_actions)
        print("")
        print("Pot: ", table.current_pot)
        print("Board: ", end="")
        Card.print_pretty_cards(table.board)
        print("")
        print("Your hand: ", end="")
        Card.print_pretty_cards(self.hand)
        print("%s \n" % (p_string))
        print("Your stack: %d" % self.stack)
        print("")
        print("Current bet: ", table.current_bet)


    def write_actions_data(self):
        self.actions_df.to_csv(self.output_path)