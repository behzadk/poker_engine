from deuces import Card
from deuces import Deck
from deuces import Evaluator

from itertools import cycle
import numpy as np
import action_policy
import matplotlib.pyplot as plt

from player import Player
from table import Table

from global_constants import * 

from itertools import combinations_with_replacement
from random import shuffle

def generate_odds():
    card_ints = [i for i in range(13)]
    pair_combos = combinations_with_replacement(card_ints, 2)
    eval = Evaluator()

    pair_scores = np.zeros(shape=(13, 13))

    pair_suits = [8, 8]
    for x in pair_combos:
        deck = Deck()
        deck_match_idxs = []
        hero_cards = [None, None]

        if x[0] == x[1]:
            continue

        # Find cards in deck
        for deck_idx, card in enumerate(deck.cards):

            if x[0] == Card.get_rank_int(card) and pair_suits[0] == Card.get_suit_int(card):
                hero_cards[0] = card
                deck_match_idxs.append(deck_idx)

            if x[1] == Card.get_rank_int(card) and pair_suits[1] == Card.get_suit_int(card):
                hero_cards[1] = card
                deck_match_idxs.append(deck_idx)

        # Remove hero cards from deck
        deck.cards = [i for idx, i in enumerate(deck.cards) if idx not in deck_match_idxs]

        # Repeat x times
        num_wins = 0
        num_losses = 0
        while num_wins + num_losses < 10000:
            # Draw villan cards
            villan_cards = deck.draw(2)

            # Draw five card board
            board = deck.draw(5)


            # Find winner
            hero_rank = eval.evaluate(hero_cards, board)
            villan_rank = eval.evaluate(villan_cards, board)

            if hero_rank < villan_rank:
                num_wins += 1

            elif hero_rank > villan_rank:
                num_losses += 1

            else:
                None

            # Put villan and board cards back into deck
            deck.cards.extend(villan_cards)
            deck.cards.extend(board)


            # Shuffle deck for next draw
            shuffle(deck.cards)


        pair_scores[x[0], x[1]] = num_wins / (num_wins + num_losses)
        pair_scores[x[1], x[0]] = num_wins / (num_wins + num_losses)
        np.savetxt('./suited_pair_scores.csv', pair_scores, delimiter=',', fmt='%5f')
        print(pair_scores)




if __name__ == "__main__":
	generate_odds()
