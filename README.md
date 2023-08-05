# deep-Q-learning-pacman
Reinforcement learning is one of the machine learning methods in which the agent learns experimentally how to act to achieve its goal by interacting with its environment. In this method, the agent receives reward or punishment by performing operations in the environment, which is used as feedback for learning. The main goal of reinforcement learning is to optimize the agent's performance in the environment.

Bellman's equation is one of the main reinforcement learning equations used to optimize the agent's performance in the environment. This equation is as follows:

Q(s.a) = Q(s.a) + α[r + γ max(Q(s^'.a')) - Q(s.a)]

In this equation, 𝑄(𝑠.𝑎) represents the value of action 𝑎 in state 𝑠 𝛼 is the learning rate that determines how much feedback at each stage should be applied to the agent's performance 𝑟 is the reward or punishment received by the agent γ is the parameter Forgetting is what determines the effect of the discount in the future. 𝑚𝑎𝑥(𝑄(𝑠′.𝑎′)) is the value of the better action in the next state 𝑠′. By using this equation, the agent can improve its performance in the environment to achieve its goal.
In reinforcement learning, the neural network is used as an approximation of the action value function (Q). By receiving inputs from the environment and actions of the agent, the neural network calculates the action value for each situation and action. Then, using the Bellman equation, the network parameters are updated in such a way that the optimal action value is obtained for each state and action.
In this process, the neural network with more experience finds the ability to predict the value of a better action, and the agent performs better in the environment using this ability.

Now, using all these algorithms in Pac-Man game, we intend to train robots (ghosts).
Our idea is to zone the playing field and consider each zone as a robot's territory. For convenience, we only do the work for one robot at first.
The robot tries to reach Pac-Man, if the distance is close, it receives positive points.
Also, for every coin that Pac-Man eats, you give the robot negative points.
Now, with these positive and negative scores, the bot's neural network learns the game.
On the other hand, we have two types of positive points, a main positive point and a secondary positive point, such that their learning coefficients are different; The main score has a higher coefficient, so it has a greater effect on learning, while the non-main score has a smaller coefficient, so it has a lower effect on learning.
In the whole step by step, all these scores are added together and the learning process is completed with the help of the Bellman equation.
In this section, we are going to see how to use the neural network and the Torch library together, and also find the right parameters for the algorithm.
![image](https://github.com/erfan-golshan/deep-Q-learning-pacman/assets/129675348/803182b2-2fad-4c2a-b0f0-a75024c83a6a)
Torch library modules all work with tensor, so we write a function that converts the desired value to tensor:
![image](https://github.com/erfan-golshan/deep-Q-learning-pacman/assets/129675348/20d794e8-c822-477d-86f1-59e9939e7897)
Next, we will write a function that will display the playground for us:
![image](https://github.com/erfan-golshan/deep-Q-learning-pacman/assets/129675348/9e512e0e-e13f-4d43-a1b1-397f74f20833)
Then we define the playing field for it:
![image](https://github.com/erfan-golshan/deep-Q-learning-pacman/assets/129675348/2650fcd4-e7e9-4f3c-b753-d62805815459)
The playing field will be like this:
Here, black places are walls and white places are empty. Also yellow houses
There are coins that Pac-Man must eat to get points.
![image](https://github.com/erfan-golshan/deep-Q-learning-pacman/assets/129675348/f2acaa8d-dda3-482d-9bcc-47235e542af7)
Now we create the neural network model and call it model:
Visually, our neural network looks like this:
![image](https://github.com/erfan-golshan/deep-Q-learning-pacman/assets/129675348/15842bc9-f8d8-4656-a1d3-fad531827f73)
The criterion function specifies how to calculate the error for us. In this neural network, the Mean Square Error model or MSE for short is used. We intend to use this function to calculate the following value in the DQN algorithm:
![image](https://github.com/erfan-golshan/deep-Q-learning-pacman/assets/129675348/ff4165e6-2b00-4753-8c02-82f4358a160f)
Since we are dealing with random movements (Pac-Man's next move is random from the bot's point of view), we need to use an optimization method based on random data. Among the various methods, SGD or Stochastic Gradient Descent method has been used, which reduces the amount of gradient (changes in the components of each neuron) to obtain the optimal value.
This SGD function takes the parameters of the neural network model and performs optimization with a learning rate and momentum. The learning rate, or lr for short, determines the value that the optimizer function needs to move towards the values ​​with a negative gradient so that the loss value is the least. Momentum value is a coefficient between 1 and 0 that is multiplied in the previous state and enters the next state. This value is usually 0.9
Now we specify the input and output values ​​of the neural network:
Inputs: current coordinates of the robot including x and y
Outputs: new coordinates of the robot including x and y and the value of loss or reward at this stage
The amount of loss or failure: the distance to Pac-Man and the amount of coins that Pac-Man has eaten since the beginning.
In fact, this neural network has feedback (according to the DQN algorithm), that is, the network can be considered as follows
![image](https://github.com/erfan-golshan/deep-Q-learning-pacman/assets/129675348/005aba32-bdc6-478c-91c5-0220ed6a6641)
You may ask why the number of coins eaten is also considered as an output and not dependent on our inputs, the answer is that the number of coins eaten also represents the elapsed time in some way, that is, the nervous system must He learns that the choices he makes need to be made in less time, otherwise the number of coins eaten will increase. Of course, instead of the number of coins eaten, we could directly specify the time parameter. But we have considered this case for testing here.
What happens is that the first output of the neural network is trained for the length component of Pac-Man, and also the second component is trained for the width component of Pac-Man. But since the bot has to move only 1 house and cannot jump on the screen, we have to train the outputs of the neural network for the direction of the bot's movement.
![image](https://github.com/erfan-golshan/deep-Q-learning-pacman/assets/129675348/8bc4127a-2fea-4df3-8765-8d5b5827e01f)
Now we come to the results review section:
The red house represents Bot and the blue house is Pac-Man.
The result is as follows
![image](https://github.com/erfan-golshan/deep-Q-learning-pacman/assets/129675348/3c2e0aa6-3aeb-4a13-9fc7-56e1190e31fe)
The red diagram is Bot's location and the blue diagram is Pac-Man's location
![image](https://github.com/erfan-golshan/deep-Q-learning-pacman/assets/129675348/97c196f5-6cfa-4c70-9a4e-63c5d0bcf803)
As can be seen from the diagram, the bot's location is closer to Pac-Man's location than it is able to catch him.
The above test was performed for different values ​​of lr and momentum, the best values ​​(in terms of convergence and in terms of time) were the values ​​mentioned earlier.
Ideas that can be used to continue the project:
- Using an optimal algorithm for Pac-Man movement
- Using another neural network for Pac-Man (finally two neural networks will compete with each other!)
- Using 3 other bots in the playground with separate neural networks
- Using a neural network connected to each other for all 4 bots (that is, for example, 4 bots are connected to each other)
- ...
