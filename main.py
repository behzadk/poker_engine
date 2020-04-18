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

# def bot_battle(checkpoint_dir_0, name_agent_0, checkpoint_dir_1, name_agent_1):
def bot_battle(checkpoint_dir_0, name_agent_0, checkpoint_dir_1, name_agent_1):
    starting_stack = 10000

    print("Loading ", checkpoint_dir_0)
    policy_0 = action_policy.RlBot(agent_name=name_agent_0, training=False, epsilon_testing=0.0, checkpoint_dir=checkpoint_dir_0)
    print("")
    print("Loading ", checkpoint_dir_1)
    policy_1 = action_policy.RlBot(agent_name=name_agent_1, training=False, epsilon_testing=0.0, checkpoint_dir=checkpoint_dir_1)

    policy_0.agent.write_state_action = False
    policy_1.agent.write_state_action = False

    player_0 = Player(name_agent_0, starting_stack, policy=policy_0.get_action)
    player_1 = Player(name_agent_1, starting_stack, policy=policy_1.get_action)

    players_list = [player_0, player_1]
    table = Table(players_list)

    hand_count = 0
    games_played = 0

    player_0_wins = 0
    player_1_wins = 0

    while games_played < 1000:
        print(hand_count)

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

            player_0.stack = starting_stack
            player_1.stack = starting_stack

            player_0.prev_stack = starting_stack
            player_1.prev_stack = starting_stack

            player_0.active = True
            player_1.active = True
            print("GAMES PLYD: ", games_played, " P0 WINS: ", player_0_wins, " P1 WINS: ", player_1_wins, "P1/P0: ", player_1_wins / games_played)

            table.prepare_next_hand()

            # for p in players_list:
            #     p.write_actions_data()

            if games_played == 10 and player_1_wins/games_played < 0.1:
                break

            if games_played == 30 and player_1_wins/games_played < 0.2:
                break

            if games_played == 100 and player_1_wins/games_played < 0.45:
                break

            if games_played == 300 and player_1_wins/games_played < 0.5:
                break

            if games_played == 500 and player_1_wins/games_played < 0.6:
                break

    print(player_1_wins / games_played)

    return (player_1_wins / games_played)

def test_model_accuracy(model):
    starting_stack = 10000
    checkpoint_dir_0 = "./checkpoint_bvb_6"
    name_agent_0 = "bvb_6"
    name_agent_1 = "hero"

    policy_0 = action_policy.RlBot(agent_name=name_agent_0, training=False, epsilon_testing=1.0, checkpoint_dir=checkpoint_dir_0)
    policy_1 = action_policy.RlBot(agent_name=name_agent_1, training=False, epsilon_testing=0.0, checkpoint_dir=checkpoint_dir_0)

    policy_1.agent.model = model

    player_0 = Player(name_agent_0, starting_stack, policy=policy_0.get_action)
    player_1 = Player(name_agent_1, starting_stack, policy=policy_1.get_action)

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

            player_0.stack = starting_stack
            player_1.stack = starting_stack

            player_0.prev_stack = starting_stack
            player_1.prev_stack = starting_stack

            player_0.active = True
            player_1.active = True
            print("GAMES PLYD: ", games_played, " P0 WINS: ", player_0_wins, " P1 WINS: ", player_1_wins, "P1/P0: ", player_1_wins / games_played)

            table.prepare_next_hand()

            # for p in players_list:
            #     p.write_actions_data()

            if games_played == 10 and player_1_wins/games_played < 0.1:
                break

            if games_played == 30 and player_1_wins/games_played < 0.2:
                break

            if games_played == 100 and player_1_wins/games_played < 0.45:
                break

            if games_played == 300 and player_1_wins/games_played < 0.5:
                break

            if games_played == 500 and player_1_wins/games_played < 0.6:
                break

    print(player_1_wins / games_played)

    return (player_1_wins / games_played)


def train_bot_vs_bot():
    starting_stack = 10000

    p0_policy = action_policy.RlBot(agent_name="bvb_8", training=False, epsilon_testing=0.0, checkpoint_dir="./checkpoint_bvb_8/")
    p1_policy = action_policy.RlBot(agent_name="bvb_9", training=True, epsilon_testing=0.1, checkpoint_dir="./checkpoint_bvb_9/")

    p0_bot = Player("p0_bot", starting_stack, policy=p0_policy.get_action)
    p1_bot = Player("p2_bot", starting_stack, policy=p1_policy.get_action)
    players_list = [p0_bot, p1_bot]
    table = Table(players_list)

    p0_stack_history = []
    p1_stack_history = []
    
    p0_stack_history.append(p0_bot.stack)
    p1_stack_history.append(p1_bot.stack)
    hand_count = 0
    games_played = 0
    
    min_max_scaling = lambda a, b, min_x, max_x, x: a + ((x - min_x) * (b - a)) / (max_x - min_x)

    for i in range(1, int(1e12)):
        hand_count += 1
        # print("")
        p0_bot.prev_stack = p0_bot.stack
        p1_bot.prev_stack = p1_bot.stack

        table.play_single_hand()
        table.prepare_next_hand()
        
        p0_stack_history.append(p0_bot.stack)
        p1_stack_history.append(p1_bot.stack)

        p1_stack_change = p1_stack_history[-1] - p1_stack_history[-2]
        normalised_hand_reward = p1_stack_history[-1] / p1_stack_history[-2]
        # max_change = p1_stack_history[-2] * 2
        # normalised_hand_reward = min_max_scaling(0, 1, 0, 2, normalised_hand_reward)
        # print(normalised_hand_reward)

        if (hand_count % 100) == 0:
            print(games_played, hand_count, p1_policy.agent.get_replay_memory_size())
        # Normalised, max gain is twice what we started with
        # minimum is to lose everything

        if p1_bot.hand_action_count > 0:
            p1_policy.agent.update_end_hand_reward(0)
            p1_bot.hand_action_count = 0

        if len(table.get_active_players()) <= 1:
            normalised_p1_reward = (p1_stack_history[-1] / starting_stack) / 2
            normalised_p1_reward = np.clip(normalised_p1_reward, 0, 1.0)
            p1_policy.agent.update_end_episode_reward(normalised_p1_reward)
            is_full = p1_policy.agent.update_replay_memory()

                # if is_full :
                #     return 0

            p1_bot.hand_action_count = 0


            p0_bot.stack = starting_stack
            p1_bot.stack = starting_stack

            p0_stack_history = []
            p1_stack_history = []

            p0_stack_history.append(p0_bot.stack)
            p1_stack_history.append(p1_bot.stack)

            p0_bot.active = True
            p1_bot.active = True
            games_played += 1

            p0_bot.hand_action_count = 0
            p1_bot.hand_action_count = 0

            table.prepare_next_hand()
            # for p in players_list:
            #     p.actions_df = p.init_actions_dataframe()



def simulate_game(checkpoint_dir_0, name_agent_0):
    starting_stack = 10000

    p1_policy = action_policy.RlBot(agent_name=name_agent_0, training=False, epsilon_testing=1.0, checkpoint_dir=checkpoint_dir_0)

    p1 = Player("HUMAN", starting_stack, policy=action_policy.huamn_action)
    rl_bot = Player("p1", starting_stack, policy=p1_policy.get_action)
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


def main():
    checkpoint_dir_0 = "./checkpoint_bvb_8/"
    name_agent_0 = "bvb_8"

    checkpoint_dir_1 = "./checkpoint_bvb_9/"
    name_agent_1 = "bvb_9"
    # train_bot_vs_bot()
    # exit()
    # simulate_game(checkpoint_dir_1, checkpoint_dir_1)

    bot_battle(checkpoint_dir_0=checkpoint_dir_0, name_agent_0=name_agent_0, 
        checkpoint_dir_1=checkpoint_dir_1, name_agent_1=name_agent_1)
    # simulate_game()

    # deck = Deck()
    # p1_hand = deck.draw(2)
    # eval = Evaluator()
    # x = eval.evaluate(p1_hand, [])

    # p2_hand = deck.draw(2)

    # board = deck.draw(5)
    # board = deck.draw(5)

    # hands = [p1_hand, p2_hand]

    # Card.print_pretty_cards(p1_hand)
    # Card.print_pretty_cards(p2_hand)

    # Card.print_pretty_cards(board)

    # evaluator = Evaluator()

    # p1_score = evaluator.evaluate(board, p1_hand)
    # p1_class = evaluator.get_rank_class(p1_score)

    # evaluator.hand_summary(hands, board)

    # print("Player 1 hand rank = %d (%s)\n" % (p1_score, evaluator.class_to_string(p1_class)))



if __name__ == "__main__":
    # checkpoint_dir_1 = "./checkpoint_bvb_1"
    # name_agent_1 = "bvb_1"

    # simulate_game(checkpoint_dir_1, name_agent_1)
    while True:
        main()
        print("Testing")
        print("Training")
        train_bot_vs_bot()



