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




def simulate_game():
    starting_stack = 1000000

    rl_bot_agent = action_policy.RlBot(training=False)


    p1 = Player("player_1", starting_stack, policy=action_policy.random_min_raise)
    p2 = Player("HUMAN", starting_stack, policy=action_policy.huamn_action)
    rl_bot = Player("rl_bot", starting_stack, policy=rl_bot_agent.get_action)
    # p4 = Player("player_4", starting_stack, policy=action_policy.random_min_raise)

    players_list = [p1, rl_bot]

    table = Table(players_list)

    p1_stack_history = []
    bot_stack_history = []

    hand_number = [0]
    p1_stack_history.append(p1.stack)
    bot_stack_history.append(rl_bot.stack)


    rl_bot_wins = 0
    p1_wins = 0
    games_played = 0

    hand_count = 0
    while games_played < 50:
    # for i in range(1, int(50)):
        table.play_single_hand()
        table.prepare_next_hand()
        hand_number.append(hand_count)
        hand_count += 1

        p1_stack_history.append(p1.stack)
        bot_stack_history.append(rl_bot.stack)

        bot_change = bot_stack_history[-1] - bot_stack_history[-2]

        # Normalised, max gain is twice what we started with
        # minimum is to lose everything
        max_gain = bot_stack_history[-2]
        max_loss = -bot_stack_history[-2]
        normalised_bot_reward = 2 * (bot_change - max_loss) / (max_gain - max_loss) - 1


        rl_bot_agent.agent.update_replay_memory(end_hand_reward=normalised_bot_reward)


        if len(table.get_active_players()) <= 1:
            print(games_played)

            if p1.stack > rl_bot.stack:
                p1_wins += 1

            else:
                rl_bot_wins += 1

            p1.stack = starting_stack
            rl_bot.stack = starting_stack

            p1.prev_stack = starting_stack
            rl_bot.prev_stack = starting_stack

            p1_stack_history.append(p1.stack)
            bot_stack_history.append(rl_bot.stack)

            p1.active = True
            rl_bot.active = True
            games_played += 1
            table.prepare_next_hand()

    print("RL_BOT_WINS: ", rl_bot_wins)
    print("p1_bot_wins: ", p1_wins)

    # for p in players_list:
    #     p.write_actions_data()

    # plt.plot(hand_number, p1_stack_history, label="P1")
    # plt.plot(hand_number, bot_stack_history, label="BOT")
    # # plt.plot(hand_number, p3_stack_history, label="RAND")
    # # plt.plot(hand_number, p4_stack_history, label="RAND")
    # plt.legend()

    # plt.show()




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



