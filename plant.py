'''
Author: Scott Underwood
Date: 02/28/2023

This script contains a class Plant which can be used 
to determine optimal operation of a wind plus storage 
system.
'''
import numpy as np
import pandas as pd

GAMMA = 0.99
ALPHA = 0.5
EPSILON = 0.3
DELTA_CHARGE = 10 # charge bins are 10 MWh apart
N_ITER = 50


class Plant:
    '''
    This class contains methods to implement Q-learning 
    for a wind plus storage plant and determine the 
    optimal action for each state in the input data.

    To create an instance of this class:
    Plant(input_data, bins_file)
    '''
    def __init__(self, input_data, bins_file):
        '''
        The __init__ method initializes the class 
        attributes, including initializing the policies 
        and value functions to appropriate values.
        '''
        self.df = pd.read_csv(input_data)
        self.df['policy'] = 0
        self.lmp_avg = self.df['LMP'].mean()
        bins = np.load(bins_file)
        self.power_bins = bins['power']
        self.lmp_bins = bins['lmp']
        self.charge_bins = bins['charge']

        self.actions = {
            0: 'Discharge battery, sell wind',
            1: 'Hold battery, sell wind',
            2: 'Charge battery from wind',
        }

        self.P = np.full([self.power_bins.size, self.lmp_bins.size, self.charge_bins.size], 1) # initialize policy space
        self.Q = np.zeros([self.power_bins.size, self.lmp_bins.size, self.charge_bins.size, len(self.actions)]) # initialize Q

        # initialize policy according to intuition (sell at high prices, charge at low)
        self.P[:,1:int(self.lmp_bins.size/3),:] = 2
        self.P[:,int(self.lmp_bins.size/3)+1:int(self.lmp_bins.size*2/3),:] = 1
        self.P[:,int(self.lmp_bins.size*2/3)+1:,:] = 0
        # make sure all edge charge states are initialized to policy moving in other direction
        self.P[:,:,0] = 2
        self.P[:,:,self.charge_bins.size-1] = 0

        # initialize Q value to be -inf for all disallowed actions
        self.Q[:,:,0,0] = float('-inf')
        self.Q[:,:,self.charge_bins.size-1,2] = float('-inf')


    def get_possible_actions(self, state):
        '''
        This method takes a state dictionary as an input 
        and returns a list of the possible actions from 
        the given state.

        Inputs:
            state (dict): current state
        
        Outputs:
            possible_actions (list): list of possible actions 
                from provided state
        '''
        charge_ind = state['charge']

        possible_actions = []

        # can discharge battery, if battery isn't depleted
        if charge_ind > 0:
            possible_actions.append(0)
        
        possible_actions.append(1) # can always hold battery level

        # can charge battery if battery isn't full (which is last state)
        if charge_ind < self.charge_bins.size-1:
            possible_actions.append(2)
        
        return possible_actions
    

    def get_next_charge(self, charge_ind, action):
        '''
        This method takes a current charge and action 
        and returns the next charge state.

        Inputs:
            charge_ind (int): current charge state
            action (int): action to be taken
        
        Outputs:
            charge_ind (int): next charge state
        '''
        # if discharging battery, reduce charge
        if action == 0:
            charge_ind -= 1
        # if holding, no change 
        elif action == 1:
            pass
        # if charging battery, increase charge
        elif action == 2:
            charge_ind += 1

        return charge_ind


    def calc_reward(self, state, action):
        '''
        This method takes a state dictionary and an 
        action as inputs and returns the reward from 
        taking the action in the provided state.

        Inputs:
            state (dict): current state
            action (int): action to be taken
        
        Outputs:
            reward (float): reward from taking action in 
                provided state
        '''
        delta_lmp = (self.lmp_avg - self.lmp_bins[state['lmp']])*1000 # kWh to MWh
        E_wind = self.power_bins[state['power']]/4 # 15 minute intervals, so divide by 4 to get MWh

        # discharge battery and sell wind, so all energy goes out
        if action == 0:
            delta_E = -DELTA_CHARGE - E_wind
        # hold battery and sell wind, so delta E is just E_wind out
        elif action == 1:
            delta_E = -E_wind
        # charge battery one full delta_charge, then sell rest of 
        # wind energy to grid
        elif action == 2:
            delta_E = DELTA_CHARGE - E_wind

        reward = delta_E*delta_lmp

        return reward


    def Q_learning(self):
        '''
        This method performs Q-learning on the dataset. This 
        involves iterating through each of the data points 
        and updating the value function using an epsilon-greedy 
        exploration policy.
        '''
        charge_ind = 4 # initialize to middle charging state

        # iterate through each row
        for index, row in self.df.iterrows():
            # update charge in dataframe
            self.df.loc[index, 'charge_binned'] = charge_ind

            lmp_ind = row['lmp_binned']
            power_ind = row['power_binned']

            state = {
                    'lmp': lmp_ind,
                    'power': power_ind,
                    'charge': charge_ind
                    }

            possible_actions = self.get_possible_actions(state)

            # epsilon greedy, so choose random if less than epsilon
            if np.random.uniform() < EPSILON:
                action = np.random.choice(possible_actions)
            # otherwise choose best action
            else:
                action = np.argmax(self.Q[power_ind,lmp_ind,charge_ind,:])
                # if action isn't allowed, choose a random allowed action
                # this shouldn't ever happen, but put this in as safeguard
                if action not in possible_actions:
                    action = np.random.choice(possible_actions)

            next_charge_ind = self.get_next_charge(charge_ind, action) # get next state
            reward = self.calc_reward(state, action) # find reward from S-A pair

            # update Q using equation 17.10
            next_lmp_ind = lmp_ind
            next_power_ind = power_ind

            next_state = {
                        'lmp': next_lmp_ind,
                        'power': next_power_ind,
                        'charge': next_charge_ind
                        }

            next_possible_actions = self.get_possible_actions(next_state)
            # find max Q from possible next actions
            next_Q = np.max(self.Q[next_power_ind,next_lmp_ind,next_charge_ind,next_possible_actions])

            # calculate new reward in equation 17.10
            value = reward + GAMMA*next_Q
            # update Q value
            self.Q[power_ind,lmp_ind,charge_ind,action] += ALPHA*(value - self.Q[power_ind,lmp_ind,charge_ind,action])

            # update policy w/ optimal action
            optimal_action = np.argmax(self.Q[power_ind,lmp_ind,charge_ind,:])
            self.P[power_ind,lmp_ind,charge_ind] = optimal_action

            # update charge state
            charge_ind = next_charge_ind


    def implement_policy(self):
        '''
        This method iterates through each data point, 
        implementing the optimal policy and adjusting 
        the next charge state accordingly.
        '''
        charge_ind = self.df.loc[0, 'charge_binned']

        for index, row in self.df.copy().iterrows():
            lmp_ind = row['lmp_binned']
            power_ind = row['power_binned']

            policy = self.P[power_ind,lmp_ind,charge_ind]
            self.df.loc[index, 'policy'] = policy

            next_charge = self.get_next_charge(charge_ind, policy)
            self.df.loc[index+1, 'charge_binned'] = next_charge
            charge_ind = next_charge


def main():
    input_file = 'state_space.csv'
    bins_file = 'bins.npz'

    P = Plant(input_file, bins_file)
    # iterate through Q-learning to convergence
    for _ in range(N_ITER):
        P.Q_learning()
    P.implement_policy()
    # output policies
    P.df.to_csv('policies.csv', index=False)


if __name__ == "__main__":
    main()
