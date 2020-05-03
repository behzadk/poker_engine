import numpy as np
import pandas as pd
from deuces import Card
from deuces import Deck
from deuces import Evaluator
from sklearn.preprocessing import OneHotEncoder
import yaml

class StateEncoder:
    def __init__(self, checkpoint_dir):
        with open(checkpoint_dir + "encode_config.yaml", 'r') as yaml_file:
            self.encoding_config = yaml.load(yaml_file, Loader=yaml.FullLoader)

        all_card_rank_values = np.array([x for x in range(14)]).reshape(14, 1)
        # All possible suit values (0 - 4 inclusive)
        # +  9 - used for board cards when they have not been dealt
        all_card_suit_values = np.array([0, 1, 2, 4, 8]).reshape(5, 1)

        self.rank_enc = OneHotEncoder(handle_unknown='error', categories='auto')
        self.rank_enc.fit(all_card_rank_values)

        self.suit_enc = OneHotEncoder(handle_unknown='error', categories='auto')
        self.suit_enc.fit(all_card_suit_values)

        deck = Deck()
        all_card_raw_values = deck.draw(52)
        all_card_raw_values.sort()
        all_card_raw_values = np.array(all_card_raw_values).reshape(52, 1)
        self.raw_card_enc = OneHotEncoder(handle_unknown='error', categories='auto')
        self.raw_card_enc.fit(all_card_raw_values)

        # Put in  dummy variables for undealt cards
        self.table_card_ranks = [0 for x in range(5)]
        self.table_card_suits = [0 for x in range(5)]

        self.evaluator = Evaluator()

        self.min_max_scaling = lambda a, b, min_x, max_x, x: a + ((x - min_x) * (b - a)) / (max_x - min_x)

        self.preflop_suited_array = np.loadtxt("./preflop_odds/suited_pair_scores.csv", delimiter=',')
        self.preflop_unsuited_array = np.loadtxt("./preflop_odds/unsuited_pair_scores.csv", delimiter=',')
        self.normalise_preflop_arrays()
    
    def get_state_shape(self):
        n_inputs = 0

        if self.encoding_config['make_hand_rank_dummies']:
            n_inputs += 26

        if self.encoding_config['make_hand_suit_dummies']:
            n_inputs += 8

        if self.encoding_config['make_board_rank_dummies']:
            n_inputs += 65

        if self.encoding_config['make_board_suit_dummies']:
            n_inputs += 20

        if self.encoding_config['make_single_vector_cards_state']:
            n_inputs += 52

        if self.encoding_config['make_stage_dummies']:
            n_inputs += 4

        if self.encoding_config['make_preflop_probability']:
            n_inputs += 1

        if self.encoding_config['make_score']:
            n_inputs += 1

        if self.encoding_config['make_hero_position']:
            n_inputs += 1

        if self.encoding_config['make_stack_prev_stack_ratio']:
            n_inputs += 1

        if self.encoding_config['make_win_stack_ratio']:
            n_inputs += 1

        if self.encoding_config['make_pot_stack_ratio']:
            n_inputs += 1

        if self.encoding_config['make_bet_stack_ratio']:
            n_inputs += 1

        if self.encoding_config['make_stack_min_bet_ratio']:
            n_inputs += 1

        if self.encoding_config['make_is_suited']:
            n_inputs += 1


        return n_inputs

        
    def normalise_preflop_arrays(self):
        max_value = np.max(self.preflop_unsuited_array)
        min_value = np.min(self.preflop_unsuited_array)

        self.preflop_suited_array = self.min_max_scaling(0, 1, min_value, max_value, self.preflop_suited_array)
        self.preflop_unsuited_array = self.min_max_scaling(0, 1, min_value, max_value, self.preflop_unsuited_array)

        self.preflop_suited_array = 1 - self.preflop_suited_array
        self.preflop_unsuited_array = 1 - self.preflop_unsuited_array


    def make_hand_rank_dummies(self, hero_player):
        card_dummies = []

        for x in hero_player.hand:
            rank_int = Card.get_rank_int(x)
            rank_int = np.array(rank_int).reshape(-1, 1)
            rank_dummies = self.rank_enc.transform(rank_int).toarray()[:,:-1]
            card_dummies.append(rank_dummies)

        
        card_dummies = np.concatenate(card_dummies, axis=1)

        return card_dummies

    def make_hand_suit_dummies(self, hero_player):
        card_dummies = []

        for x in hero_player.hand:
            suit_int = Card.get_suit_int(x)
            suit_int = np.array(suit_int).reshape(-1, 1)
            suit_dummies = self.suit_enc.transform(suit_int).toarray()[:,:-1]
            card_dummies.append(suit_dummies)

        card_dummies = np.concatenate(card_dummies, axis=1)

        return card_dummies

    def make_board_rank_dummies(self, table):
        card_dummies = []

        table_card_ranks = self.table_card_ranks[:]

        for idx, x in enumerate(table.board):
            rank_int = Card.get_rank_int(x)
            table_card_ranks[idx] = rank_int

        for x in table_card_ranks:
            x = np.array(x).reshape(-1, 1)
            rank_dummies = self.rank_enc.transform(x).toarray()[:,:-1]
            card_dummies.append(rank_dummies)

        card_dummies = np.concatenate(card_dummies, axis=1)

        return card_dummies

    def make_board_suit_dummies(self, table):
        card_dummies = []
        table_card_suits = self.table_card_suits[:]

        for idx, x in enumerate(table.board):
            suit_int = Card.get_suit_int(x)
            table_card_suits[idx] = suit_int

        for x in table_card_suits:
            x = np.array(x).reshape(-1, 1)
            suit_dummies = self.suit_enc.transform(x).toarray()[:,:-1]
            card_dummies.append(suit_dummies)
        
        card_dummies = np.concatenate(card_dummies, axis=1)

        return card_dummies

    def make_single_vector_cards_state(self, hero_player, table):
        hand = hero_player.hand
        board = [x for x in table.board if x != 13]

        cards = hand + board

        cards = np.array(cards).reshape(-1, 1)

        single_enc_vec = self.raw_card_enc.transform(cards).toarray()
        state = np.sum(single_enc_vec, axis=0).reshape(1, -1)

        return state



    def make_stage_dummies(self, table):
        is_preflop = 0
        is_flop = 0
        is_turn = 0
        is_river = 0

        if table.stage == "PREFLOP":
            pass

        elif table.stage == "FLOP":
            is_flop = 1

        elif table.stage == "TURN":
            is_turn = 1

        elif table.stage == "RIVER":
            is_river = 1

        stage_dummies = np.array([is_preflop, is_flop, is_turn, is_river]).reshape(1, -1)

        return(stage_dummies)

    def make_preflop_probability(self, hero_player, table):
        if table.stage == "PREFLOP":
            card_0_rank = Card.get_rank_int(hero_player.hand[0])
            card_0_suit = Card.get_suit_int(hero_player.hand[0])

            card_1_rank = Card.get_rank_int(hero_player.hand[1])
            card_1_suit = Card.get_suit_int(hero_player.hand[1])

            if card_0_suit == card_1_suit:
                hero_score = self.preflop_suited_array[card_0_rank, card_1_rank]

            else:
                hero_score = self.preflop_unsuited_array[card_0_rank, card_1_rank]

        else:
            hero_score = -1

        return np.array(hero_score).reshape(1, -1)

    def make_score(self, hero_player, table):
        if table.stage == "PREFLOP":
            is_preflop = 1

        else:
            is_preflop = 0

        hero_score = 7462
        # evaluate hand if not preflop
        if not is_preflop:
            board = [x for x in table.board if x != 13]
            hero_score = self.evaluator.evaluate(board, hero_player.hand)
            hero_score = self.min_max_scaling(0, 1, 0, 7462, hero_score)            

        else:
            hero_score = -1.0

        return np.array(hero_score).reshape(1, -1)

    def make_hero_position(self, hero_player):
        return np.array(hero_player.playing_position).reshape(1, -1)

    def make_stack_prev_stack_ratio(self, hero_player):
        stack_contrib_ratio = hero_player.stack / hero_player.prev_stack
        return np.array(stack_contrib_ratio).reshape(1, -1)

    def make_win_stack_ratio(self, hero_player, table):
        win_stack_ratio = (hero_player.stack + table.current_pot) / hero_player.prev_stack
        return np.array(win_stack_ratio).reshape(1, -1)

    def make_stack_min_bet_ratio(self, hero_player, table):
        stack_min_bet_ratio = (table.big_blind) / hero_player.stack
        return np.array(stack_min_bet_ratio).reshape(1, -1)

    def make_bet_stack_ratio(self, hero_player, table):
        bet_stack_ratio = table.current_bet / hero_player.stack
        return np.array(bet_stack_ratio).reshape(1, -1)

    def make_pot_stack_ratio(self, hero_player, table):
        make_pot_stack_ratio = table.current_pot / hero_player.stack

        return np.array(make_pot_stack_ratio).reshape(1, -1)


    def make_is_suited(self, hero_player):
        if Card.get_suit_int(hero_player.hand[0]) == Card.get_suit_int(hero_player.hand[1]):
            return np.array(1).reshape(1, -1)

        else:
            return np.array(0).reshape(1, -1)


    def encode_state(self, hero_player, table):
        state = []

        if self.encoding_config['make_hand_rank_dummies']:
            state.append(self.make_hand_rank_dummies(hero_player))

        if self.encoding_config['make_hand_suit_dummies']:
            state.append(self.make_hand_suit_dummies(hero_player))

        if self.encoding_config['make_board_rank_dummies']:
            state.append(self.make_board_rank_dummies(table))

        if self.encoding_config['make_board_suit_dummies']:
            state.append(self.make_board_suit_dummies(table))

        if self.encoding_config['make_single_vector_cards_state']:
            state.append(self.make_single_vector_cards_state(hero_player, table))

        if self.encoding_config['make_stage_dummies']:
            state.append(self.make_stage_dummies(table))

        if self.encoding_config['make_preflop_probability']:
            state.append(self.make_preflop_probability(hero_player, table))

        if self.encoding_config['make_score']:
            state.append(self.make_score(hero_player, table))

        if self.encoding_config['make_hero_position']:
            state.append(self.make_hero_position(hero_player))

        if self.encoding_config['make_stack_prev_stack_ratio']:
            state.append(self.make_stack_prev_stack_ratio(hero_player))

        if self.encoding_config['make_win_stack_ratio']:
            state.append(self.make_win_stack_ratio(hero_player, table))

        if self.encoding_config['make_bet_stack_ratio']:
            state.append(self.make_bet_stack_ratio(hero_player, table))

        if self.encoding_config['make_pot_stack_ratio']:
            state.append(self.make_pot_stack_ratio(hero_player, table))

        if self.encoding_config['make_stack_min_bet_ratio']:
            state.append(self.make_stack_min_bet_ratio(hero_player, table))

        if self.encoding_config['make_is_suited']:
            state.append(self.make_is_suited(hero_player))


        state = np.concatenate(state, axis=1).reshape(1, -1)

        return state


if __name__ == "__main__":
    deck = Deck()
