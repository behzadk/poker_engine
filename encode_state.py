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
		all_card_rank_values = np.array([x for x in range(14)]).reshape(14, 1)
		# All possible suit values (0 - 4 inclusive)
		# +  9 - used for board cards when they have not been dealt
		all_card_suit_values = np.array([1, 2, 4, 8, 9]).reshape(5, 1)


		self.rank_enc = OneHotEncoder(handle_unknown='error', categories='auto')
		self.rank_enc.fit(all_card_rank_values)

		self.suit_enc = OneHotEncoder(handle_unknown='error', categories='auto')
		self.suit_enc.fit(all_card_suit_values)

		# Put in  dummy variables for undealt cards
		self.table_card_ranks = [13 for x in range(5)]
		self.table_card_suits = [9 for x in range(5)]

		# test = np.array(13).reshape(-1, 1)
		# out = self.rank_enc.transform(test).toarray()[:,:-1]
		# print(out)
		# exit()

	def get_card_dummies(self, rank_int, suit_int):		
		rank_int = np.array(rank_int).reshape(-1, 1)
		suit_int = np.array(suit_int).reshape(-1, 1)

		rank_dummies = self.rank_enc.transform(rank_int).toarray()[:,:-1]
		suit_dummies = self.suit_enc.transform(suit_int).toarray()[:,:-1]
		
		ret = np.concatenate([rank_dummies, suit_dummies], axis=1)
		# ret = np.concatenate([rank_int, suit_dummies], axis=1)

		return ret

	def encode_state_v1(self, table, hero_player):

		state_arrays = []
		for x in hero_player.hand:
			card_0_rank = Card.get_rank_int(x)
			card_0_suit = Card.get_suit_int(x)
			state_arrays.append(self.get_card_dummies(card_0_rank, card_0_suit))

		state = np.concatenate([state_arrays], axis=1).reshape(1, 1, -1)
		return state


		# Put in  dummy variables for undealt cards
		table_card_ranks = self.table_card_ranks[:]
		table_card_suits = self.table_card_suits[:]

		for idx, x in enumerate(table.board):
			table_card_ranks[idx] = Card.get_rank_int(x)
			table_card_suits[idx] = Card.get_suit_int(x)

		for card_rank, card_suit in zip(table_card_ranks, table_card_suits):
			state_arrays.append(self.get_card_dummies(card_rank, card_suit))

		state_arrays = np.concatenate(state_arrays, axis=1)

		bet_and_stack = []
		bet_and_stack.append(np.array(hero_player.stack/table.total_chips).reshape(1, -1))
		bet_and_stack.append(np.array(table.current_bet/table.total_chips).reshape(1, -1))
		bet_and_stack = np.concatenate(bet_and_stack, axis=1)

		state = np.concatenate([state_arrays, bet_and_stack], axis=1).reshape(1, 1, -1)

		return state


	def encode_state_simple(self, table, hero_player):
		state_arrays = [1]

		for x in hero_player.hand:
			card_0_rank = (float(Card.get_rank_int(x)) + 1) / 13
			state_arrays.append(card_0_rank)

		if Card.get_suit_int(hero_player.hand[0]) == Card.get_suit_int(hero_player.hand[1]):
			is_suited = 1

		else:
			is_suited = 0

		state_arrays.append(is_suited)

		card_connectivity = abs(Card.get_rank_int(hero_player.hand[0]) - Card.get_rank_int(hero_player.hand[1])) ** 0.25
		state_arrays.append(card_connectivity)

		# state = np.concatenate([state_arrays], axis=1).reshape(1, 1, -1)
		state_arrays = np.array(state_arrays).reshape(1, -1)

		bet_and_stack = []
		bet_and_stack.append(np.array(table.current_pot/hero_player.prev_stack).reshape(1, -1))
		bet_and_stack.append(np.array(hero_player.stack/hero_player.prev_stack).reshape(1, -1))
		bet_and_stack.append(np.array(hero_player.playing_position).reshape(1, -1))


		bet_and_stack = np.concatenate(bet_and_stack, axis=1)

		state = np.concatenate([state_arrays, bet_and_stack], axis=1).reshape(1, -1)

		fold_state = np.copy(state)
		fold_state[0][0:7] = 0.0

		return state, fold_state


if __name__ == "__main__":
	state_encoder()
	# encode_state_v1(0, 1)


