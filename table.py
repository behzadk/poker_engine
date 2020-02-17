from deuces import Card
from deuces import Deck
from deuces import Evaluator
from global_constants import * 
import pandas as pd

class Table:
    def __init__(self, players_list):
        self.hand_idx = 0

        self.current_stage = 'PREFLOP'
        self.current_pot = 0
        self.current_bet = 0

        self.players = players_list

        self.total_chips = sum([p.stack for p in self.players])

        self.d_chip_pos = 0

        self.stage = None
        self.big_blind = 20
        self.small_blind = 10
        self.ante = 10

        self.deck = Deck()
        self.board = []

        self.reset_player_position_indexes()

        self.eval = Evaluator()

    ##
    # Deals two cards to each player
    ##
    def deal_hole_cards(self):
        for p in self.players:
            p.hand = self.deck.draw(2)
            p.hand = sorted(p.hand, reverse=False)

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
        for p in self.players:
            p.evaluate_hand(self.board, self.eval)

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

                if p.active:
                    player_action = p.get_action(self, self, round_actions)
                    self.current_pot += player_action[2]

                    if player_action[2] != 0:
                        self.current_bet = player_action[2]

                    if player_action[1] == "RAISE":
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
        self.players[0].contribution_to_pot += self.small_blind

        # Take big blind
        self.players[1].stack -= self.big_blind
        self.current_pot += self.big_blind
        self.players[1].contribution_to_pot += self.big_blind

        # Take ante
        for p in self.players:
            p.stack  -= self.ante
            self.current_pot += self.ante
            p.contribution_to_pot += self.ante


    ##
    # Returns a list of winners from active players
    # and the current board.
    ##
    def identify_winners(self, players):
        player_hands = [p.hand for p in players]        
        best_rank = 7463  # rank one worse than worst hand
        winners = []

        for p in players:
            p_rank = self.eval.evaluate(p.hand, self.board)

            if p_rank == best_rank:
                winners.append(p)

            elif p_rank < best_rank:
                best_rank = p_rank
                winners = [p]

        return winners

    
    def redistribute_pot(self):
        # Sort players by their contribution to the pot
        all_players = sorted(self.players, key=lambda p:p.contribution_to_pot, reverse=False)

        making_side_pots = True

        pots = []

        for p_1 in all_players:
            this_pot_val = 0
            this_pot_players = []

            player_contrib = p_1.contribution_to_pot

            # Subtract the player contribution for all players if balance > 0
            for p_2 in all_players:
                if p_2.contribution_to_pot > 0:
                    p_2.contribution_to_pot -= player_contrib
                    this_pot_val += player_contrib
                    this_pot_players.append(p_2)

            # Add pot
            if this_pot_val > 0:
                pots.append([this_pot_val, this_pot_players])

        # For each pot compare the hands of active players to identify the winner
        active_players = self.get_active_players()
        active_player_hands = [p.hand for p in active_players]
        best_rank = 7463  # rank one worse than worst hand

        # For each pot compare the hands of active players to identify the winner
        for pot in pots:
            pot_members = pot[1]
            active_pot_members = [p for p in pot_members if p in active_players]

            pot_winners = []

            # No active winners
            if len(active_pot_members) == 0:
                pot_winners = pot_members

            # Find the winners in the pot
            else:
                pot_winners = self.identify_winners(active_pot_members)


            # Split pot equally beween the winners
            distributed_pot_val = pot[0] / len(pot_winners)

            for winner in pot_winners:
                winner.stack += distributed_pot_val


    ##
    # Resets the table and moves dealerchip to the left by one position.
    # 
    ##
    def prepare_next_hand(self, update_action_data=False):
        self.board = []
        self.current_pot = 0

        # Reorder players by moving first to back
        player_0 = self.players[0]
        self.players.pop(0)
        self.players.append(player_0)
        self.reset_player_position_indexes()

        # Set players with no stack to inactive
        for p in self.players:
            if p.stack <= 0:
                p.active = False

            else:
                p.active = True

        # self.players = [p for p in self.players if p.active]

        # Remove player hole cards
        for p in self.players:
            if update_action_data:
                p.update_action_data_net_stack(self.hand_idx)
                
            p.hand = None
            p.stage_actions = []
            p.hand_actions = []
            p.prev_stack = p.stack

        # Reset the deck
        self.deck = Deck()
        self.hand_idx += 1


    def play_single_hand(self):
        stages = ['PREFLOP', 'FLOP', 'TURN', 'RIVER']

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

        # self.players[0].display_game_state(self, [])
        # Cleanup by allocating chips to winner
        self.redistribute_pot()

