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

def bot_battle(checkpoint_dir_0, name_agent_0, checkpoint_dir_1, name_agent_1):
    starting_stack = 10000

    print("Loading ", checkpoint_dir_0)
    policy_0 = action_policy.RlBot(agent_name=name_agent_0, training=False, epsilon_testing=1.0, checkpoint_dir=checkpoint_dir_0)
    print("")
    print("Loading ", checkpoint_dir_1)
    policy_1 = action_policy.RlBot(agent_name=name_agent_1, training=False, epsilon_testing=0.0, checkpoint_dir=checkpoint_dir_1)

    player_0 = Player("p0", starting_stack, policy=policy_0.get_action)
    player_1 = Player("p1", starting_stack, policy=policy_1.get_action)

    players_list = [player_0, player_1]
    table = Table(players_list)

    hand_count = 0
    games_played = 0

    player_0_wins = 0
    player_1_wins = 0

    while games_played < 1000:
        table.play_single_hand()
        table.prepare_next_hand()
        hand_count += 1

        # Normalised, max gain is twice what we started with
        # minimum is to lose everything        
        if len(table.get_active_players()) <= 1:
            if player_0.stack > player_1.stack:
                player_0_wins += 1
            
            else:
                player_1_wins += 1
                
            games_played += 1
            print("GAMES PLYD: ", games_played, " P0 WINS: ", player_0_wins, " P1 WINS: ", player_1_wins)
            print(player_1_wins / games_played)

            player_0.stack = starting_stack
            player_1.stack = starting_stack

            player_0.prev_stack = starting_stack
            player_1.prev_stack = starting_stack

            player_0.active = True
            player_1.active = True

            table.prepare_next_hand()

            for p in players_list:
                p.write_actions_data()

def train_bot_vs_bot():
    starting_stack = 10000
    # x = np.arange(0, 1.1, 0.01)
    # y = []

    n=2
    norm_reward_hill = lambda x, k: x**n / (k**n + x**n + 1e-12)

    p0_policy = action_policy.RlBot(agent_name="bvb_0", training=True, epsilon_testing=0.1, checkpoint_dir="./checkpoint_bvb_0")
    p1_policy = action_policy.RlBot(agent_name="bvb_1", training=True, epsilon_testing=0.1, checkpoint_dir="./checkpoint_bvb_1")

    p0_bot = Player("p0_bot", starting_stack, policy=p0_policy.get_action)
    p1_bot = Player("p1_bot", starting_stack, policy=p1_policy.get_action)

    players_list = [p0_bot, p1_bot]
    table = Table(players_list)

    p0_stack_history = []
    p1_stack_history = []
    
    p0_stack_history.append(p0_bot.stack)
    p1_stack_history.append(p1_bot.stack)
    hand_count = 0
    games_played = 0

    for i in range(1, int(1e12)):
        # print("")
        table.play_single_hand()
        table.prepare_next_hand()
        # print(hand_count)
        
        hand_count += 1

        p0_stack_history.append(p0_bot.stack)
        p1_stack_history.append(p1_bot.stack)

        p0_stack_change = p0_stack_history[-1] - p0_stack_history[-2]
        p1_stack_change = p1_stack_history[-1] - p1_stack_history[-2]

        # Normalised, max gain is twice what we started with
        # minimum is to lose everything
        max_gain = p0_stack_history[-2]
        max_loss = -p0_stack_history[-2]
        p0_stack_change = p0_stack_history[-1] - p0_stack_history[-2]
        # p0_stack_change = np.clip(p0_stack_change, 0,a_max=None)
        normalised_p0_reward = ( p0_stack_change - max_loss) / (max_gain - max_loss)
        normalised_p0_reward = norm_reward_hill(normalised_p0_reward, 0.5)
        # print(p0_stack_change, normalised_p0_reward)
        # print(normalised_p0_reward)
        p0_policy.agent.update_replay_memory(end_hand_reward=normalised_p0_reward)


        max_gain = p1_stack_history[-2]
        max_loss = -p1_stack_history[-2]
        p1_stack_change = p1_stack_history[-1] - p1_stack_history[-2]

        normalised_p1_reward = ( p1_stack_change - max_loss) / (max_gain - max_loss)
        normalised_p1_reward = norm_reward_hill(normalised_p1_reward, 0.5)
        p1_policy.agent.update_replay_memory(end_hand_reward=normalised_p1_reward)
        
        if len(table.get_active_players()) <= 1:
            if (hand_count % 100) == 0:
                print(games_played, hand_count)

            p0_bot.stack = starting_stack
            p1_bot.stack = starting_stack

            p0_bot.prev_stack = starting_stack
            p1_bot.prev_stack = starting_stack

            p0_stack_history = []
            p1_stack_history = []

            p0_stack_history.append(p0_bot.stack)
            p1_stack_history.append(p1_bot.stack)

            p0_bot.active = True
            p1_bot.active = True
            games_played += 1
            table.prepare_next_hand()
            for p in players_list:
                p.actions_df = p.init_actions_dataframe()



def simulate_game():
    starting_stack = 10000

    rl_bot_agent = action_policy.RlBot(training=True, checkpoint_dir="./checkpoint")

    p1 = Player("player_1", starting_stack, policy=action_policy.random_min_raise)
    p2 = Player("HUMAN", starting_stack, policy=action_policy.huamn_action)
    rl_bot = Player("rl_bot", starting_stack, policy=rl_bot_agent.get_action)
    # p4 = Player("player_4", starting_stack, policy=action_policy.random_min_raise)

    players_list = [p1, rl_bot]

    table = Table(players_list)

    p1_stack_history = []
    bot_stack_history = []

    p1_stack_history.append(p1.stack)
    bot_stack_history.append(rl_bot.stack)


    rl_bot_wins = 0
    p1_wins = 0
    games_played = 0

    hand_count = 0
    # while games_played < 50:
    for i in range(1, int(1e12)):
        table.play_single_hand()
        table.prepare_next_hand()
        hand_count += 1

        p1_stack_history.append(p1.stack)
        bot_stack_history.append(rl_bot.stack)

        bot_change = bot_stack_history[-1] - bot_stack_history[-2]

        # Normalised, max gain is twice what we started with
        # minimum is to lose everything
        max_gain = bot_stack_history[-2]
        max_loss = -bot_stack_history[-2]
        normalised_bot_reward = 2 * (bot_change - max_loss) / (max_gain - max_loss) - 1

        # rl_bot_agent.agent.update_replay_memory(end_hand_reward=normalised_bot_reward)

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
    checkpoint_dir_0 = "./checkpoint_bvb_0"
    name_agent_0 = "bvb_0"

    checkpoint_dir_1 = "./checkpoint_bvb_1"
    name_agent_1 = "bvb_1"
    # train_bot_vs_bot()
    # exit()

    bot_battle(checkpoint_dir_0=checkpoint_dir_0, name_agent_0=name_agent_0, 
        checkpoint_dir_1=checkpoint_dir_1, name_agent_1=name_agent_1)
    # simulate_game()

    deck = Deck()
    p1_hand = deck.draw(2)
    eval = Evaluator()
    x = eval.evaluate(p1_hand, [])

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
    train_bot_vs_bot()



