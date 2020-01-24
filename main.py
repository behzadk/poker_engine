from treys import Card
from treys import Deck
from treys import Evaluator

from itertools import cycle
import numpy as np
import action_policy
import matplotlib.pyplot as plt

from player import Player
from table import Table

from global_constants import * 




def simulate_game():
    starting_stack = 10000

    p1 = Player("player_1", starting_stack, policy=action_policy.random_min_raise)
    p2 = Player("player_2", starting_stack, policy=action_policy.random_min_raise)
    # p3 = Player("player_3", starting_stack, policy=action_policy.random_min_raise)
    # p4 = Player("player_4", starting_stack, policy=action_policy.random_min_raise)

    players_list = [p1, p2]

    table = Table(players_list)

    p1_stack_history = []
    p2_stack_history = []
    p3_stack_history = []
    p4_stack_history = []

    hand_number = [0]
    p1_stack_history.append(p1.stack)
    p2_stack_history.append(p2.stack)

    print("playing game")
    for i in range(1, 5000):
        table.play_single_hand()
        table.prepare_next_hand()
        hand_number.append(i)

        p1_stack_history.append(p1.stack)
        p2_stack_history.append(p2.stack)
        # p3_stack_history.append(p3.stack)
        # p4_stack_history.append(p4.stack)

        # print(p1.stack + p2.stack)

        if len(table.get_active_players()) == 1:
            break


    for p in players_list:
        p.write_actions_data()

    plt.plot(hand_number, p1_stack_history, label="RAISER")
    plt.plot(hand_number, p2_stack_history, label="RAISER")
    # plt.plot(hand_number, p3_stack_history, label="RAND")
    # plt.plot(hand_number, p4_stack_history, label="RAND")
    plt.legend()

    plt.show()




def main():
    simulate_game()
    exit()


    deck = Deck()
    p1_hand = deck.draw(2)
    p2_hand = deck.draw(2)

    board = deck.draw(5)
    board = deck.draw(5)


    hands = [p1_hand, p2_hand]

    Card.print_pretty_cards(p1_hand)
    Card.print_pretty_cards(p2_hand)

    Card.print_pretty_cards(board)

    evaluator = Evaluator()

    p1_score = evaluator.evaluate(board, p1_hand)
    p1_class = evaluator.get_rank_class(p1_score)

    evaluator.hand_summary(hands, board)

    # print("Player 1 hand rank = %d (%s)\n" % (p1_score, evaluator.class_to_string(p1_class)))



if __name__ == "__main__":
    main()



