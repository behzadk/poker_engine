import numpy as np
import pandas as pd
from deuces import Card
from deuces import Deck
from deuces import Evaluator
from sklearn.preprocessing import OneHotEncoder

class StateEncoder:
    def __init__(self):
        # All possible card ranks (0-12 inclusive) 
        # +  13 - used for board cards when they have not been dealt
        
        # all_card_rank_values = np.array([x for x in range(14)]).reshape(14, 1)
        # All possible suit values (0 - 4 inclusive)
        # +  9 - used for board cards when they have not been dealt
        all_card_suit_values = np.array([1, 2, 4, 8, 9]).reshape(5, 1)


        self.rank_enc = OneHotEncoder(handle_unknown='error', categories='auto')
        self.rank_enc.fit(all_card_rank_values)

        self.suit_enc = OneHotEncoder(handle_unknown='error', categories='auto')
        self.suit_enc.fit(all_card_suit_values)

        # Put in  dummy variables for undealt cards
        self.table_card_ranks = [0 for x in range(5)]
        self.table_card_suits = [0 for x in range(5)]

        self.evaluator = Evaluator()

        self.min_max_scaling = lambda a, b, min_x, max_x, x: a + ((x - min_x) * (b - a)) / (max_x - min_x)


        self.preflop_suited_array = np.loadtxt("./preflop_odds/suited_pair_scores.csv", delimiter=',')
        self.preflop_unsuited_array = np.loadtxt("./preflop_odds/unsuited_pair_scores.csv", delimiter=',')
        self.normalise_preflop_arrays()


        # test = np.array(13).reshape(-1, 1)
        # out = self.rank_enc.transform(test).toarray()[:,:-1]
        # print(out)
        # exit()

    def normalise_preflop_arrays(self):
        max_value = np.max(self.preflop_unsuited_array)
        min_value = np.min(self.preflop_unsuited_array)
        self.preflop_suited_array = self.min_max_scaling(0, 1, min_value, max_value, self.preflop_suited_array)
        self.preflop_unsuited_array = self.min_max_scaling(0, 1, min_value, max_value, self.preflop_unsuited_array)

        self.preflop_suited_array = 1 - self.preflop_suited_array
        self.preflop_unsuited_array = 1 - self.preflop_unsuited_array

    def get_card_dummies(self, rank_int, suit_int):
        rank_int = np.array(rank_int).reshape(-1, 1)
        suit_int = np.array(suit_int).reshape(-1, 1)

        rank_dummies = self.rank_enc.transform(rank_int).toarray()[:,:-1]
        suit_dummies = self.suit_enc.transform(suit_int).toarray()[:,:-1]
        
        ret = np.concatenate([rank_dummies, suit_dummies], axis=1)
        # ret = np.concatenate([rank_int, suit_dummies], axis=1)

        return ret

    def get_suit_dummies(self, suit_int):
        suit_int = np.array(suit_int).reshape(-1, 1)
        suit_dummies = self.suit_enc.transform(suit_int).toarray()[:,:-1]
        ret = np.concatenate([suit_dummies], axis=1)


    def encode_state_v1(self, table, hero_player):
        state_arrays = []
        for x in hero_player.hand:
            card_0_rank = Card.get_rank_int(x)
            card_0_suit = Card.get_suit_int(x)
            state_arrays.append(self.get_card_dummies(card_0_rank, card_0_suit))

        state = np.concatenate([state_arrays], axis=1).reshape(1, 1, -1)

        # Put in  dummy variables for undealt cards
        table_card_ranks = self.table_card_ranks[:]
        table_card_suits = self.table_card_suits[:]

        for idx, x in enumerate(table.board):
            table_card_ranks[idx] = Card.get_rank_int(x)
            table_card_suits[idx] = Card.get_suit_int(x)


        print(table.board)
        print(table_card_ranks)
        print(table_card_ranks == self.table_card_ranks)
        for card_rank, card_suit in zip(table_card_ranks, table_card_suits):
            state_arrays.append(self.get_card_dummies(card_rank, card_suit))

        state_arrays = np.concatenate(state_arrays, axis=1)

        bet_and_stack = []

        bet_and_stack.append(np.array(hero_player.playing_position).reshape(1, -1))
        bet_and_stack.append(np.array(table.current_pot/table.total_chips).reshape(1, -1))
        bet_and_stack.append(np.array(hero_player.stack/table.total_chips).reshape(1, -1))
        bet_and_stack.append(np.array(table.current_bet/table.total_chips).reshape(1, -1))
        bet_and_stack.append(np.array(hero_player.stack/table.big_blind).reshape(1, -1))

        bet_and_stack = np.concatenate(bet_and_stack, axis=1)

        state = np.concatenate([state_arrays, bet_and_stack], axis=1).reshape(1, -1)

        fold_state = np.copy(state)
        # fold_state = np.zeros(shape=np.shape(state))


        return state, fold_state

    def encode_state_by_hand(self, table, hero_player):
        state_arrays = []

        # Unpack card ranks
        for x in hero_player.hand:
            # card_0_rank = (float(Card.get_rank_int(x)) + 1) / 13
            card_0_rank = Card.get_rank_int(x)
            card_0_rank = self.min_max_scaling(-1, 1, 0, 12, card_0_rank)
            state_arrays.append(card_0_rank)

        c_1_suit = Card.get_suit_int(hero_player.hand[0])
        # c_1_suit = self.min_max_scaling(-1, 1, 0, 9, c_1_suit)
        c_2_suit = Card.get_suit_int(hero_player.hand[1])
        # c_2_suit = self.min_max_scaling(-1, 1, 0, 9, c_2_suit)

        # state_arrays.append(c_1_suit)
        # state_arrays.append(c_2_suit)

        if c_1_suit == c_2_suit:
            state_arrays.append(1)

        else:
            state_arrays.append(-1)

        # state = np.concatenate([state_arrays], axis=1).reshape(1, 1, -1)


        # is preflop
        is_preflop = 0
        is_flop = 0
        is_turn = 0
        is_river = 0

        if table.stage == "PREFLOP":
            is_preflop = 1
        elif table.stage == "FLOP":
            is_flop = 1
        elif table.stage == "TURN":
            is_turn = 1
        elif table.stage == "RIVER":
            is_river = 1

        else:
            print("error")
            exit()

        state_arrays.append(is_preflop)

        hero_score = 7462
        # evaluate hand if not preflop
        if not is_preflop:
            board = [x for x in table.board if x != 13]
            hero_score = self.evaluator.evaluate(board, hero_player.hand)
            hero_score = self.min_max_scaling(0, 1, 0, 7462, hero_score)

        else:
            card_0_rank = Card.get_rank_int(hero_player.hand[0])
            card_1_rank = Card.get_rank_int(hero_player.hand[1])

            if c_1_suit == c_2_suit:
                hero_score = self.preflop_suited_array[card_0_rank, card_1_rank]

            else:
                hero_score = self.preflop_unsuited_array[card_0_rank, card_1_rank]

            # if hero_score == 0:
            #     print(c_1_suit, c_2_suit)
            #     print(card_0_rank, card_1_rank)
            #     print(self.preflop_unsuited_array)
            #     exit()

        state_arrays.append(hero_score)

        state_arrays = np.array(state_arrays)
        state_arrays = state_arrays.reshape(1, -1)

        bet_and_stack = []
        bet_and_stack.append(np.array(hero_player.playing_position).reshape(1, -1))

        # pot_total_ratio = table.current_pot/table.total_chips
        # stack_total_ratio = hero_player.stack/table.total_chips
        bet_total_ratio = table.current_bet/table.total_chips
        # stack_bb_ratio = table.big_blind/hero_player.stack
        stack_contrib_ratio = hero_player.stack / hero_player.prev_stack
        win_stack_ratio = (hero_player.stack + table.current_pot) / hero_player.prev_stack

        # bet_total_ratio = self.min_max_scaling(-1, 1, 0, 1, bet_total_ratio)
        # stack_contrib_ratio = self.min_max_scaling(-1, 1, 0, 2, stack_contrib_ratio)
        # win_stack_ratio = self.min_max_scaling(-1, 1, 1, 2, win_stack_ratio)

        # bet_and_stack.append(np.array(pot_total_ratio).reshape(1, -1))
        # bet_and_stack.append(np.array(stack_total_ratio).reshape(1, -1))
        bet_and_stack.append(np.array(bet_total_ratio).reshape(1, -1))
        # bet_and_stack.append(np.array(stack_bb_ratio).reshape(1, -1))
        bet_and_stack.append(np.array(stack_contrib_ratio).reshape(1, -1))
        bet_and_stack.append(np.array(win_stack_ratio).reshape(1, -1))

        # bet_and_stack.append(np.array(table.current_pot/table.total_chips).reshape(1, -1))
        # bet_and_stack.append(np.array(hero_player.stack/table.total_chips).reshape(1, -1))
        # bet_and_stack.append(np.array(table.current_bet/table.total_chips).reshape(1, -1))
        # bet_and_stack.append(np.array(hero_player.stack/table.big_blind).reshape(1, -1))
        # bet_and_stack.append(np.array(hero_player.stack/hero_player.contribution_to_pot).reshape(1, -1))

        bet_and_stack = np.concatenate(bet_and_stack, axis=1)

        # print(bet_and_stack)

        state = np.concatenate([state_arrays, bet_and_stack], axis=1).reshape(1, -1)

        # fold_state = np.copy(state)
        fold_state = np.zeros(shape=np.shape(state))
        fold_state[-1][-1] = stack_contrib_ratio

        # Fold state doesn't matter what the hero cards are, score, position

        return state, fold_state


    def encode_state_by_table(self, table, hero_player):
        state_arrays = []

        # Unpack card ranks
        for x in hero_player.hand:
            # card_0_rank = (float(Card.get_rank_int(x)) + 1) / 13
            card_0_rank = Card.get_rank_int(x) + 1
            card_0_rank = self.min_max_scaling(-1, 1, 0, 13, card_0_rank)
            state_arrays.append(card_0_rank)

        c_1_suit = Card.get_suit_int(hero_player.hand[0])
        c_1_suit = self.min_max_scaling(-1, 1, 0, 9, c_1_suit)
        c_2_suit = Card.get_suit_int(hero_player.hand[1])
        c_2_suit = self.min_max_scaling(-1, 1, 0, 9, c_2_suit)

        state_arrays.append(c_1_suit)
        state_arrays.append(c_2_suit)

        # state = np.concatenate([state_arrays], axis=1).reshape(1, 1, -1)

        # Put in  dummy variables for undealt cards
        table_card_ranks = self.table_card_ranks[:]
        table_card_suits = self.table_card_suits[:]

        for idx, x in enumerate(table.board):
            table_card_ranks[idx] = Card.get_rank_int(x) + 1
            table_card_ranks[idx] = self.min_max_scaling(-1, 1, 0, 13, table_card_ranks[idx])

            table_card_suits[idx] = Card.get_suit_int(x)
            table_card_suits[idx] = self.min_max_scaling(-1, 1, 0, 9, table_card_suits[idx])

        # is preflop
        is_preflop = 0
        is_flop = 0
        is_turn = 0
        is_river = 0

        if table.stage == "PREFLOP":
            is_preflop = 1
        elif table.stage == "FLOP":
            is_flop = 1
        elif table.stage == "TURN":
            is_turn = 1
        elif table.stage == "RIVER":
            is_river = 1

        else:
            print("error")
            exit()

        state_arrays.append(is_preflop)
        state_arrays.append(is_flop)
        state_arrays.append(is_turn)
        state_arrays.append(is_river)

        hero_score = 7462
        # evaluate hand if not preflop
        if not is_preflop:
            board = [x for x in table.board if x != 13]
            hero_score = self.evaluator.evaluate(board, hero_player.hand)

        hero_score = self.min_max_scaling(-1, 1, 0, 7462, hero_score)


        state_arrays.append(hero_score)

        state_arrays = np.array(state_arrays)
        state_arrays = state_arrays.reshape(1, -1)


        bet_and_stack = []
        bet_and_stack.append(np.array(hero_player.playing_position).reshape(1, -1))

        bet_total_ratio = table.current_bet/table.total_chips
        stack_contrib_ratio = hero_player.stack / table.total_chips
        win_stack_ratio = (hero_player.stack + table.current_pot) / table.total_chips

        bet_total_ratio = self.min_max_scaling(-1, 1, 0, 1, bet_total_ratio)
        stack_contrib_ratio = self.min_max_scaling(-1, 1, 0, 2, stack_contrib_ratio)
        win_stack_ratio = self.min_max_scaling(-1, 1, 1, 2, win_stack_ratio)

        bet_and_stack.append(np.array(bet_total_ratio).reshape(1, -1))
        bet_and_stack.append(np.array(stack_contrib_ratio).reshape(1, -1))
        bet_and_stack.append(np.array(win_stack_ratio).reshape(1, -1))

        bet_and_stack = np.concatenate(bet_and_stack, axis=1)

        state = np.concatenate([state_arrays, bet_and_stack], axis=1).reshape(1, -1)

        # fold_state = np.copy(state)
        fold_state = np.zeros(shape=np.shape(state))
        fold_state[-1][-1] = stack_contrib_ratio

        # Fold state doesn't matter what the hero cards are, score, position

        return state, fold_state

    def encode_state_simple(self, table, hero_player):
        state_arrays = []

        for x in hero_player.hand:
            card_0_rank = (float(Card.get_rank_int(x)) + 1) / 13
            state_arrays.append(card_0_rank)

        if Card.get_suit_int(hero_player.hand[0]) == Card.get_suit_int(hero_player.hand[1]):
            is_suited = 1

        else:
            is_suited = 0

        state_arrays.append(is_suited)

        card_connectivity = abs(Card.get_rank_int(hero_player.hand[0]) / 13 - Card.get_rank_int(hero_player.hand[1]) / 13) ** 0.25
        state_arrays.append(card_connectivity)

        # state = np.concatenate([state_arrays], axis=1)
        state_arrays = np.array(state_arrays).reshape(1, -1)
        # print(state_arrays)

        bet_and_stack = []
        bet_and_stack.append(np.array(table.current_pot/hero_player.prev_stack).reshape(1, -1))
        bet_and_stack.append(np.array(hero_player.stack/hero_player.prev_stack).reshape(1, -1))
        
        bet_and_stack.append(np.array(hero_player.playing_position).reshape(1, -1))
        bet_and_stack = np.concatenate(bet_and_stack, axis=1)

        state = np.concatenate([state_arrays, bet_and_stack], axis=1).reshape(1, -1)

        fold_state = np.copy(state)
        fold_state[0][0:6] = 0.0

        fold_q_value = hero_player.stack/hero_player.prev_stack

        return state, fold_state


if __name__ == "__main__":
    state_encoder()
    # encode_state_v1(0, 1)


