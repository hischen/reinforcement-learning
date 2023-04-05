# reinforcement-learning
 Reinforcement learning algorithm implements.



### Tips

#### New gym API:Gym v0.21->Gymnasium
 In this repo, I use the [new API](https://gymnasium.farama.org/) for creating the env because a temporary wrapper support is provided for the old code and it may cease to be backward compatible some day. Using the new API could have certain minor ramifications to env code (in one line - Dont simply do: done = truncated).

 Gym v21 to v26 Migration Guide: https://gymnasium.farama.org/content/migration-guide

 Since Gym will not be receiving any future updates, this repo switch over to Gymnasium(import gymnasium as gym).  If you'd like to read more about the story behind this switch, please check out this [blog post](https://farama.org/Announcing-The-Farama-Foundation).   

Let us quickly understand the change.

To use the new API, add new_step_api=True option for e.g.

```
env = gym.make('MountainCar-v0', new_step_api=True)

```

This causes the env.step() method to return five items instead of four. What is this extra one?

- Well, in the old API - done was returned as True if episode ends in any way.
- In the n[requirements.txt](requirements.txt)ew API, done is split into 2 parts:
- terminated=True if environment terminates (eg. due to task completion, failure etc.)
- truncated=True if episode truncates due to a time limit or a reason that is not defined as part of the task MDP.

This is done to remove the ambiguity in the `done` signal. `done=True` in the old API did not distinguish between the environment terminating & the episode truncating. This problem was avoided previously by setting `info['TimeLimit.truncated']` in case of a timelimit through the TimeLimit wrapper. All that is not required now and the env.step() function returns us:

```
next_state, reward, terminated, truncated , info = env.step(action)

```

How could this impact your code: If your game has some kind of max_steps or timeout, you should read the 'truncated' variable IN ADDITION to the 'terminated' variable to see if your game ended. Based on the kind of rewards that you have you may want to tweak things slightly. A simplest option could just be to do a

```
done = truncated OR terminated

```
and then proceed to reuse your old code.

##  References
    
* 动手学强化学习:https://hrl.boyuai.com/chapter/intro
* 蘑菇书EasyRL:https://datawhalechina.github.io/easy-rl/#/
* https://github.com/Allenpandas/Tutorial4RL
* https://github.com/wangshusen/DRL
* https://github.com/Phoenix-Shen/ReinforcementLearning
* https://github.com/DeepRLChinese/DeepRL-Chinese


    
   
