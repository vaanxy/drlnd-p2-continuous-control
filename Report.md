[//]: #	"Image References"
[image1]: ./images/ddpg.png	"DDPG Scores"
# Report

In this project, An Actor-Critic method called Deep Deterministic Policy Gradient(DDPG) has been implemented to solve the continuous control problem.

This report  will describe this learning algorithm, along with  the model architectures for neural networks and the chosen hyperparameters.

## Deep Deterministic Policy Gradient(DDPG) 

### Actor Neural Network Architecture

The actor network mapping state to action

- Input Layer: 33
- Hidden Layer 1: 512
- Hidden Layer 2: 256
- Output Layer: 4

```python
Actor(
  (hidden_layers): ModuleList(
    (0): Linear(in_features=33, out_features=512, bias=True)
    (1): Linear(in_features=512, out_features=256, bias=True)
  )
  (output): Linear(in_features=256, out_features=4, bias=True)
)
```



### Critic Neural Network Architecture

The critoc network mapping (state, action) pair to Q-value

- Input Layer: 33
- Hidden Layer 1: 512 + 4
- Hidden Layer 2: 256
- Output Layer: 1

~~~python
Critic(
  (hidden_layers): ModuleList(
    (0): Linear(in_features=33, out_features=512, bias=True)
    (1): Linear(in_features=516, out_features=256, bias=True)
  )
  (output): Linear(in_features=256, out_features=1, bias=True)
)
~~~



### Hyper-parameters

- Replay Memory Size = 1e5
- Batch Size = 256
- GAMMA = 0.99
- TAU = 1e-3
- Actor Learning Rate = 1e-3
- Critic Learning Rate = 1e-4
- Noise Decaying Rate = 0.99

### Plot of Rewards

![DDPG Scores][image1]

DDPG solved the problem in 199 episodes.


## Future Improvement

- Fine tuning hyper parameters to get better performance;
- Try to make the network deeper to  get better performance;
- Try other Policy Gradient Algorithms, such as PPO, A3C;
