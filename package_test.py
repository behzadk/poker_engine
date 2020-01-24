import numpy as np
from table import Table
from player import Player
import action_policy
from treys import Card
from treys import Deck
from treys import Evaluator

def test_pot_redistribution():
	# Make four  players
    p1 = Player("player_1", 0, policy=action_policy.always_raise)
    p2 = Player("player_2", 0, policy=action_policy.always_raise)
    p3 = Player("player_3", 0, policy=action_policy.random_min_raise)
    p4 = Player("player_4", 0, policy=action_policy.random_min_raise)

    players_list = [p1, p2, p3, p4]

    table = Table(players_list)

    # Board gives all players royal flush
    table.board = [
    	Card.new('Ah'), 
    	Card.new('Kh'),
    	Card.new('Qh'), 
    	Card.new('Jh'),
    	Card.new('Th')
    ]


    # Set player hands
    p1.hand = [Card.new('7h'), Card.new('8h')]
    p2.hand = [Card.new('6h'), Card.new('3h')]
    p3.hand = [Card.new('5h'), Card.new('2h')]
    p4.hand = [Card.new('4h'), Card.new('9h')]

    p1.contribution_to_pot = 100
    p1.active = False
    p2.contribution_to_pot = 100
    p3.contribution_to_pot = 30
    p4.contribution_to_pot = 20

    table.redistribute_pot()

    for p in players_list:
    	print(p.stack)



if __name__ == "__main__":
	test_pot_redistribution()
