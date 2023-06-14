This is the v2 of PPO using LSTM --- the stack frame is in shape of (4, 336, 336, 1) which means stack size is in the beginning instead of being used as channels

the input for neural network is (batch size, 4, 336, 336, 1)

This is ok for sample action and make esitimation in collecting experient data

However, in training the policy, larget batch (all steps in an epoch) will be fed to network at once, which will run out the allocated memory

Hence, we need to split/yield the batch data into mini-batch, like 1-5
