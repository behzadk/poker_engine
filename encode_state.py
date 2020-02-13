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


		self.rank_enc = OneHotEncoder(handle_unknown='error', drop='first')
		self.rank_enc.fit(all_card_rank_values)

		self.suit_enc = OneHotEncoder(handle_unknown='error', drop='first')
		self.suit_enc.fit(all_card_suit_values)

		# rank_enc = OneHotEncoder(handle_unknown='error', drop='first')
		# rank_enc.fit(all_card_rank_values)
		# test = np.array(0).reshape(-1, 1)
		# out = enc.transform(test).toarray()

	# from sklearn.preprocessing import OneHotEncoder
	def get_card_dummies(self, rank_int, suit_int):		
		rank_int = np.array(rank_int).reshape(-1, 1)
		suit_int = np.array(suit_int).reshape(-1, 1)

		rank_dummies = self.rank_enc.transform(rank_int).toarray()
		suit_dummies = self.suit_enc.transform(suit_int).toarray()
		
		ret = np.concatenate([rank_dummies, suit_dummies], axis=1)

		return ret



	def encode_state_v1(self, table, hero_player):

		state_arrays = []
		for x in hero_player.hand:
			card_0_rank = Card.get_rank_int(x)
			card_0_suit = Card.get_suit_int(x)
			state_arrays.append(self.get_card_dummies(card_0_rank, card_0_suit))


		# Put in  dummy variables for undealt cards
		table_card_ranks = [13 for x in range(5)]
		table_card_suits = [9 for x in range(5)]

		for idx, x in enumerate(table.board):
			table_card_ranks[idx] = Card.get_rank_int(x)
			table_card_suits[idx] = Card.get_suit_int(x)

		for card_rank, card_suit in zip(table_card_ranks, table_card_suits):
			state_arrays.append(self.get_card_dummies(card_rank, card_suit))


		state = np.concatenate(state_arrays).reshape(1, 1, -1)

		return state





if __name__ == "__main__":
	state_encoder()

	# encode_state_v1(0, 1)