from deuces import Card
from deuces import Deck
from deuces import Evaluator

from itertools import cycle
import numpy as np
import action_policy
import matplotlib.pyplot as plt

debug = "SHOW_ACTIONS"
# debug = None

class Table:
    def __init__(self, players_list):
        self.current_stage = 'PREFLOP'
        self.current_pot = 0
        self.current_bet = 0

        self.players = players_list

        self.d_chip_pos = 0

        self.stage = None
        self.big_blind = 20
        self.small_blind = 10
        self.ante = 0

        self.deck = Deck()
        self.board = []

        self.reset_player_position_indexes()

    ##
    # Deals two cards to each player
    ##
    def deal_hole_cards(self):
        for p in self.players:
            p.hand = self.deck.draw(2)


    def generate_table_state(self):
        table_state = {
        'stage': self.stage,
        'big_blind': self.big_blind,
        'small_blind': self.small_blind,
        'd_chip_pos': self.d_chip_pos,
        'board': self.board,
        'current_bet': self.current_bet, 
        }

        return table_state

    def reset_player_position_indexes(self):
        for idx, p in enumerate(self.players):
            p.playing_position = idx

    ##
    # Sets the hand rank for each player based
    # on the players current hand and board
    ##
    def evaluate_player_hands(self):
        evaluator = Evaluator()
        for p in self.players:
            p.evaluate_hand(self.board, evaluator)

    ##
    # Returns the stack of each player
    ##
    def get_player_stacks(self):
        stacks = []

        for p in self.players:
            stacks.append(p.stack)

        return stacks

    def get_active_players(self):
        active_players = [p for p in self.players if p.active]
        return active_players


    def request_player_actions(self):
        self.current_bet = 0
        
        if len(self.get_active_players()) <= 1:
            no_further_action = True

        else:
            no_further_action = False

        stage_actions = []
        last_raiser = None

        while no_further_action is False:
            round_actions = []

            for p in self.get_active_players():
                if p == last_raiser:
                    no_further_action = True
                    break

                table_state = self.generate_table_state()

                if p.active:
                    player_action = p.get_action(table_state)
                    self.current_pot += player_action[1]

                    if player_action[1] != 0:
                        self.current_bet = player_action[1]

                    if player_action[0] == "RAISE":
                        last_raiser = p

                    round_actions.append(player_action)


            stage_actions.append(round_actions)

            if last_raiser == None:
                no_further_action = True

        for p in self.get_active_players():
            p.hand_actions += stage_actions



        if debug == "SHOW_ACTIONS":
            print(stage_actions)

    def take_blinds_and_ante(self):
        ## TODO check success of taking blinds

        # Take small blind
        self.players[0].stack -= self.small_blind
        self.current_pot += self.small_blind

        # Take big blind
        self.players[1].stack -= self.big_blind
        self.current_pot += self.big_blind

        # Take ante
        for p in self.players:
            p.stack  -= self.ante
            self.current_pot += self.ante


    ##
    # Returns a list of winners from active players
    # and the current board.
    ##
    def identify_winner(self):
        active_players = self.get_active_players()
        active_player_hands = [p.hand for p in active_players]
        evaluator = Evaluator()
        
        best_rank = 7463  # rank one worse than worst hand
        winners = []

        for p in (active_players):
            p_rank = evaluator.evaluate(p.hand, self.board)

            if p_rank == best_rank:
                winners.append(p.playing_position)

            elif p_rank < best_rank:
                best_rank = p_rank
                winners = [p.playing_position]
        print("winners")
        print(winners)
        return winners

    
    def redistribute_pot(self, winners):
        # Equal distribution between winners
        divided_pot = self.current_pot / len(winners)

        winning_players = [p for p in self.players if p.playing_position in winners]


        winning_players = sorted(winning_players, key=lambda p:p.contribution_to_pot, reverse=True)
        
        all_players_contributing = [p for p in self.players if p.contribution_to_pot > 0]
        all_players_contributing = sorted(players_contributing, key=lambda p:p.contribution_to_pot, reverse=True)
        x = sum([p.contribution_to_pot for p in players_contributing])
        print(x)

        taken_out = 0
        while len(players_contributing) != 0:
            max_contrib = players_contributing[0].contribution_to_pot

            for idx, p in enumerate(players_contributing):
                p.contribution_to_pot -= max_contrib
                p.stack += max_contrib

            players_contributing = [p for p in self.players if p.contribution_to_pot > 0]
            players_contributing = sorted(players_contributing, key=lambda p:p.contribution_to_pot, reverse=True)
        
        x = sum([p.contribution_to_pot for p in winning_players])
        print(x)

    ##
    # Resets the table and moves dealerchip to the left by one position.
    # 
    ##
    def prepare_next_hand(self):
        self.board = []
        self.current_pot = 0

        # Reorder players by moving first to back
        player_0 = self.players[0]
        self.players.pop(0)
        self.players.append(player_0)

        # Set players with no stack to inactive
        for p in self.players:
            if p.stack <= 0:
                p.active = False

            else:
                p.active = True

        self.players = [p for p in self.players if p.active]

        # Remove player hole cards
        for p in self.players:
            p.hand = None
            p.stage_actions = []
            p.hand_actions = []


        # Reset the deck
        self.deck = Deck()


    def play_single_hand(self):
        stages = ['PREFLOP', 'FLOP', 'TURN', 'RIVER']
        print("")
        ## Preflop
        self.stage = stages[0]
        self.deal_hole_cards()
        self.take_blinds_and_ante()
        self.request_player_actions()

        if debug == "SHOW_ACTIONS":
            print(self.stage, "POT: ", self.current_pot)
            print("")

        ## Flop
        self.stage = stages[1]
        self.board += self.deck.draw(3)
        self.request_player_actions()

        if debug == "SHOW_ACTIONS":
            print(self.stage, "POT: ", self.current_pot)
            print("")

        # River
        self.stage = stages[2]
        self.board += [self.deck.draw(1)]
        self.request_player_actions()
        if debug == "SHOW_ACTIONS":
            print(self.stage, "POT: ", self.current_pot)
            print("")

        # Turn
        self.stage = stages[3]
        self.board += [self.deck.draw(1)]
        self.request_player_actions()
        
        if debug == "SHOW_ACTIONS":
            print(self.stage, "POT: ", self.current_pot)
            print("")

        # Cleanup by allocating chips to winner
        winners = self.identify_winner()
        self.redistribute_pot(winners)


class Player:
    def __init__(self, player_id, starting_stack, policy):
        self.id = player_id
        self.hand = None
        self.hand_actions = []
        self.active = True

        self.stack = starting_stack
        self.hand_rank = None
        self.contribution_to_pot = 0

        self.all_in = False

        # Playing position is relative to dealer chip. 
        # dealer chip is index [-1]
        self.playing_position = None
        self.policy = policy

        self.stage_actions = []
        self.hand_actions

    ##
    # evaluates player hand
    ##
    def evaluate_hand(self, board, eval_obj):
        self.hand_rank = eval_obj.evaluate(board, self.hand)


    ##
    # Returns an action committed by a player.
    # actions are chosen from an action_policy script.
    ##
    def get_action(self, table_state):
        curent_bet = table_state['current_bet']

        if self.stack <= 0:
            self.all_in = True
            chosen_action = ['ALL_IN', 0]

        else:
            chosen_action = self.policy(table_state, self)
            
            self.stack -= chosen_action[1]


            if chosen_action[0] == 'FOLD':
                self.active = False

        self.stage_actions.append(chosen_action)
        self.contribution_to_pot += chosen_action[1]

        return chosen_action


def simulate_game():
    starting_stack = 10000

    p1 = Player("player_1", starting_stack, policy=action_policy.always_raise)
    p2 = Player("player_2", starting_stack, policy=action_policy.always_raise)
    p3 = Player("player_3", starting_stack, policy=action_policy.random_min_raise)
    p4 = Player("player_4", starting_stack, policy=action_policy.random_min_raise)

    players_list = [p1, p2, p3, p4]

    table = Table(players_list)

    p1_stack_history = []
    p2_stack_history = []
    p3_stack_history = []
    p4_stack_history = []

    hand_number = [0]
    p1_stack_history.append(p1.stack)
    p2_stack_history.append(p2.stack)
    p3_stack_history.append(p3.stack)
    p4_stack_history.append(p4.stack)

    for i in range(1, 100):
        table.play_single_hand()
        table.prepare_next_hand()
        hand_number.append(i)

        p1_stack_history.append(p1.stack)
        p2_stack_history.append(p2.stack)
        p3_stack_history.append(p3.stack)
        p4_stack_history.append(p4.stack)

        if len(table.get_active_players()) == 1:
            break

    plt.plot(hand_number, p1_stack_history, label="RAISER")
    plt.plot(hand_number, p2_stack_history, label="RAISER")
    plt.plot(hand_number, p3_stack_history, label="RAND")
    plt.plot(hand_number, p4_stack_history, label="RAND")
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



