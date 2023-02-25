# Spring 2023, 535515 Reinforcement Learning
# HW0: Policy Iteration and Value iteration for MDPs
       
import numpy as np
import gym

def get_rewards_and_transitions_from_env(env):
    # Get state and action space sizes
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    # Intiailize matrices
    R = np.zeros((num_states, num_actions, num_states))
    P = np.zeros((num_states, num_actions, num_states))

    # Get rewards and transition probabilitites for all transitions from an OpenAI gym environment
    for s in range(num_states):
        for a in range(num_actions):
            for transition in env.P[s][a]:
                prob, s_, r, done = transition
                R[s, a, s_] = r
                P[s, a, s_] = prob
                
    return R, P

def value_iteration(env, gamma=0.9, max_iterations=10**6, eps=10**-3):
    """        
        Run value iteration (You probably need no more than 30 lines)
        
        Input Arguments
        ----------
            env: 
                the target environment
            gamma: float
                the discount factor for rewards
            max_iterations: int
                maximum number of iterations for value iteration
            eps: float
                for the termination criterion of value iteration 
        ----------
        
        Output
        ----------
            policy: np.array of size (500,)
        ----------
        
        TODOs
        ----------
            1. Initialize the value function V(s)
            2. Get transition probabilities and reward function from the gym env
            3. Iterate and improve V(s) using the Bellman optimality operator
            4. Derive the optimal policy using V(s)
        ----------
    """
    num_spaces = env.observation_space.n
    num_actions = env.action_space.n
    
    # Initialize with a random policy
    policy = np.array([env.action_space.sample() for _ in range(num_spaces)])
    
    ##### FINISH TODOS HERE #####
    R, P = get_rewards_and_transitions_from_env(env)
    V = np.zeros(num_spaces)

    for i in range(max_iterations):
        V_old = V.copy()
        for s in range(num_spaces):
            max_V_new = -float('inf')
            max_a = -1
            for a in range(num_actions):
                V_new = 0
                for s2 in range(num_spaces):
                    V_new += P[s, a, s2] * (R[s, a, s2] + gamma * V[s2])

                if V_new > max_V_new:
                    max_V_new = V_new
                    max_a = a
                
            V[s] = max_V_new
            policy[s] = max_a
        
        dis = abs(np.sum(V) - np.sum(V_old))
        #print(f"iter {i}: {dis}")
        if dis < eps:
            break

    #############################
    
    # Return optimal policy    
    return policy

def policy_iteration(env, gamma=0.9, max_iterations=10**6, eps=10**-3):
    """ 
        Run policy iteration (You probably need no more than 30 lines)
        
        Input Arguments
        ----------
            env: 
                the target environment
            gamma: float
                the discount factor for rewards
            max_iterations: int
                maximum number of iterations for the policy evalaution in policy iteration
            eps: float
                for the termination criterion of policy evaluation 
        ----------  
        
        Output
        ----------
            policy: np.array of size (500,)
        ----------
        
        TODOs
        ----------
            1. Initialize with a random policy and initial value function
            2. Get transition probabilities and reward function from the gym env
            3. Iterate and improve the policy
        ----------
    """
    num_spaces = env.observation_space.n
    num_actions = env.action_space.n
    
    # Initialize with a random policy
    #policy = np.array([env.action_space.sample() for _ in range(num_spaces)])
    
    policy = np.ones([num_spaces, num_actions]) / num_actions  #initialize policy with same prob.
    
    '''
    policy = np.zeros([num_spaces, num_actions])
    for state in range(num_spaces):
        policy[state]=np.eye(num_actions)[env.action_space.sample()]
    '''
    
    ##### FINISH TODOS HERE #####
    R, P = get_rewards_and_transitions_from_env(env)
    V = np.zeros(num_spaces)
    while True:
        # Iterative Policy Evaluation
        #print("IPE:")
        for i in range(max_iterations):
            V_old = V.copy()
            for s in range(num_spaces):
                tmp = 0
                for a in range(num_actions):
                    for s2 in range(num_spaces):
                        tmp += policy[s, a] * P[s, a, s2] * (R[s, a, s2] + gamma * V[s2])

                V[s] = tmp
            
            dis = abs(np.sum(V) - np.sum(V_old))
            #print(f"iter {i}: {dis}")
            if dis < eps:
                break
        
        # one-step policy improvement
        policy_old = policy.copy()
        
        for s in range(num_spaces):
            max_a = -1
            max_Q = -float('inf')
            for a in range(num_actions):
                Q = 0
                for s2 in range(num_spaces):
                    Q += P[s, a, s2] * (R[s, a, s2] + gamma * V[s2])

                if Q > max_Q:
                    max_Q = Q
                    max_a = a
            
            tmp = np.zeros(num_actions)
            tmp[max_a] = 1
            policy[s] = tmp
        

        if (policy == policy_old).all():
            break
        
    policy = np.argmax(policy, axis=1)
    
    #############################
    
    # Return optimal policy
    return policy

def print_policy(policy, mapping=None, shape=(0,)):
    print(np.array([mapping[action] for action in policy]).reshape(shape))


def run_pi_and_vi(env_name):
    """ 
        Enforce policy iteration and value iteration
    """    
    env = gym.make(env_name)
    print('== {} =='.format(env_name))
    print('# of actions:', env.action_space.n)
    print('# of states:', env.observation_space.n)
    print(env.desc)

    vi_policy = value_iteration(env)
    pi_policy = policy_iteration(env)
    

    return pi_policy, vi_policy


if __name__ == '__main__':
    # OpenAI gym environment: Taxi-v2 or Taxi-v3
    pi_policy, vi_policy = run_pi_and_vi('Taxi-v3')

    # For debugging
    action_map = {0: "S", 1: "N", 2: "E", 3: "W", 4: "P", 5: "D"}
    print_policy(pi_policy, action_map, shape=None)
    print_policy(vi_policy, action_map, shape=None)
    
    # Compare the policies obatined via policy iteration and value iteration
    diff = sum([abs(x-y) for x, y in zip(pi_policy.flatten(), vi_policy.flatten())])        
    print('Discrepancy:', diff)
    



