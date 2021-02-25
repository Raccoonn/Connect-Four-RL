
from dqn import Agent
from environment import ConnectFour
from training_tools import *




if __name__ == '__main__':

    train = True
    load = False

    env = ConnectFour()

    filename = 'p1.h5'

    agent, memory = setup_Agent(filename)

    if load == True:
        agent.load_model(filename)
        print('\n\n... Model Loaded ...\n\n')



    batch_size = 512
    frame_skips = 5000


    wins = [0, 0, 0]
    p1_wins = []
    p2_wins = []
    draws = []


    p_i, p_syms = 0, ('\\', '|', '/', '-')

    for episode in range(1, 50000):
        env.reset()
        done = False

        state = env.board

        while not env.done:
            print('Playing a game...  ' + p_syms[p_i], end='\r')
            p_i = (p_i+1) % 4



            # Choose an action depending on player, loop until valid move selected
            # If valid options are chosen repeatedly, a random valid move
            # maybe add negative reward as well?

            tries = 0
            while True:
                if tries > 500:
                    action = random_Agent(state)
                else:

                    if env.player == 1:
                        action = agent.choose_action(state.flatten())
                    else:
                        action = random_Agent(state)

                    valid, state_, reward, winner = env.Step(action)

                if valid == True:
                    break
                else:
                    tries += 1

            
            if env.player == 1 or done == True:
                memory.store_transition(state.flatten(), action, reward, state_.flatten(), done)

            if train == True and memory.mem_cntr > frame_skips:
                agent.learn(batch_size, memory.sample_buffer(batch_size))




            state = state_

            env.update_Players()






        # Update winner tallies
        if winner == 1:
            wins[0] += 1
        elif winner == 2:
            wins[1] += 1
        else:
            wins[2] += 1

        p1_wins.append(wins[0])
        p2_wins.append(wins[1])
        draws.append(wins[2])

        plot_progress(episode, p1_wins, p2_wins, draws)


        if episode % 10 == 0 and train == True:
            agent.save_model()
            print('\n... Model Saved ...\n')







