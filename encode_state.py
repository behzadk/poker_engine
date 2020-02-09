import numpy as np
import pandas as pd
from deuces import Card
from deuces import Deck
from deuces import Evaluator
from sklearn.preprocessing import OneHotEncoder

class state_encoder:
	def __init__(self):
		all_cards = Deck().GetFullDeck()

# from sklearn.preprocessing import OneHotEncoder
def get_card_dummies(rank_int, suit_int):
	# Heart
	if suit_int == 2:
		enc_suit = [0, 0, 0]

	# Club
	if suit_int == 8:
		enc_suit = [1, 0, 0]

	# Diamond
	if suit_int == 1:
		enc_suit = [0, 1, 0]

	# Spade
	if suit_int == 4:
		enc_suit = [0, 0, 1]

	return [rank_int] + enc_suit



def encode_state_v1(table, hero_player):

	state = []
	for x in hero_player.hand:
		card_0_rank = Card.get_rank_int(x)
		card_0_suit = Card.get_suit_int(x)
		state += get_card_dummies(card_0_rank, card_0_suit)

	state = np.array(state).reshape(1, 1, -1)

	return state





if __name__ == "__main__":
	state_encoder()

	# encode_state_v1(0, 1)