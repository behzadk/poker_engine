import numpy as np

def main():
	num_used = 5
	rewards = [0, 0, 0, 0, 1]

	discount_factor = 0.01
	q_values = [0.5, 0.5, 0.5, 0.5, 0.5]

	av_calc = lambda r, q: r + discount_factor * q

	for i in range(10):

		new_q_values = [0, 0, 0, 0, 0]
		for k in reversed(range(num_used-1)):
			state_q  = q_values[k]
			next_state_q = q_values[k+1]
			state_reward = rewards[k]

			av_state = av_calc(state_reward, next_state_q)

			est_error = (av_state - state_q) * 0.0001

			q_values[k] = av_state

		print(q_values)





if __name__ == "__main__":
	main()