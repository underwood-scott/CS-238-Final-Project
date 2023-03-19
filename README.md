# Wind Plus Storage System Optimization
This repository contains scripts that use reinforcement
learning to determine the optimal policy for any given 
state in a wind plus storage system. The input data 
files come from the Electric Reliability Council of 
Texas (ERCOT).These are formatted using 
`data_processing.py` and then output into 
`state_space.csv`. Then, `plant.py` uses Q-learning to 
find the optimal policy for each data point, and 
outputs the optimal policy to `policies.csv`. Lastly, 
`results_analysis.py` reads in the policies and compares 
the performance of the policy to a baseline of a system 
with no storage included.
