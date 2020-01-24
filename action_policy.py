import numpy as np

## Action format:
# ['ACTION TYPE', int(value to transfer to pot)]
# e.g ['RAISE', 10], ['FOLD', 0]
# 
def random_min_raise(current_table_state, this_player):
	current_bet = current_table_state['current_bet']
	min_raise = current_table_state['big_blind']


	action_type_space = ['CALL', 'RAISE', 'CHECK', 'FOLD']

	# No bets made, can only check or raise
	if current_bet == 0:
		action_type_space = ['RAISE', 'CHECK']

	elif current_bet > this_player.stack:
		action_type_space = ['CALL', 'FOLD']


	else:
		action_type_space = ['CALL', 'RAISE', 'FOLD']

	chosen_action_type = np.random.choice(action_type_space)

	action_value = 0

	# If call, value to transfer is current bet
	if chosen_action_type == 'CALL':
		if current_bet > this_player.stack:
			action_value = this_player.stack

		else:
			action_value = current_bet

	# If raise, apply min raise
	elif chosen_action_type == 'RAISE':
		if min_raise + current_bet > this_player.stack:
			min_raise = this_player.stack

		action_value = current_bet + min_raise

	# If fold, value to transfer is 0
	elif chosen_action_type == 'FOLD':
		action_value = 0

	# If check, value to transfer is 0
	elif chosen_action_type == 'CHECK':
		action_value = 0

	else:
		print("Invalid action type, quitting... ")
		exit()

	ret_action = [chosen_action_type, action_value]

	return ret_action


def always_raise(current_table_state, this_player):
	current_bet = current_table_state['current_bet']
	min_raise = current_table_state['big_blind']

	max_raise = this_player.stack - current_bet

	action_type_space = ['CALL', 'RAISE', 'CHECK', 'FOLD']

	if this_player.stack < max_raise:
		chosen_action_type = 'CALL'
		action_value = current_bet

	else:
		chosen_action_type = 'RAISE'
		action_value = current_bet + np.random.uniform(min_raise, max_raise)

	ret_action = [chosen_action_type, action_value]

	return ret_action

