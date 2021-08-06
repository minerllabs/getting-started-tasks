# Tasks for getting started with MineRL

This repository contains examples and small tasks on getting
started with MineRL environment.

To begin, install the [requirements for MineRL](https://minerl.readthedocs.io/en/latest/tutorials/index.html),
and then install Python requirements with with `pip install -r requirements.txt`. If you encounter
problems with installation, you can still complete these tasks with Colab links provided. 

If you have any questions, you can reach us on [Discord](https://discord.com/invite/BT9uegr).
If you spot typos/bugs in any of the tasks or this repo, do tell us via Github issues!

## Tasks

Stars indicate the difficulty of the task. Click the task to see more details.

<details>
  <summary>:star: Getting started with MineRL</summary>
  Start by playing bit of Minecraft via MineRL with `play_with_minerl.py` script.
  After this checkout `getting_familiar_with_minerl_and_gym.py` to get a feeling
  how to control the agent.
  
  You can find the latter task on Colab [here](https://colab.research.google.com/drive/11CVCeb7f0P2nqcgWGLG1wDZcE3AxngxL?usp=sharing).
</details>

<details>
  <summary>:star: Improve Intro baseline for the Diamond competition</summary>
  A step-by-step instructions to improve a simple, fully-scripted agent for obtaining
  wood and stone in the MineRLObtainDiamond-v0 task. Start out by [opening this document](https://docs.google.com/document/d/12d0jMnsoR5xjyye4Rlpo84yJOZRMbfSYOb17OWOJdFw/edit)
  and following the instructions.
</details>

<details>
  <summary>:star: :star: Implementing behavioural cloning from (almost) scratch.</summary>
  Start by opening up `behavioural_cloning.py` and following the instructions at the beginning of the file in comments (fill in missing code pieces).
  After completion and training you should have an ok-ish agent that can obtain logs.
  You can also find the task on Colab [here](https://colab.research.google.com/drive/1JQ9suwMe-TnyBoDjhdydI6Ic35-m6NLh?usp=sharing).
  
  You can find a crude reference answers [in this Colab notebook](https://colab.research.google.com/drive/1JQ9suwMe-TnyBoDjhdydI6Ic35-m6NLh?usp=sharing).
  This task is built on the [BC + scripted baseline solution](https://github.com/KarolisRam/MineRL2021-Intro-baselines/blob/main/standalone/BC_plus_script.py).
</details>

<details>
  <summary>:star: :star: :star: Learn how to use stable-baselines and imitation libraries with MineRL.</summary>
  This walk-through demonstrates how to combine well-established reinforcement learning (stable-baselines3) and imitation learning (imitation) libraries
  with MineRL to train more sophisticated agents. Start by opening [this Colab link](https://colab.research.google.com/drive/13_jI8YLk9ATRQSd7_3rV5rOsll7jsSz0),
</details>

<details>
  <summary>:star: :star: :star: Improve Research baseline for the Diamond competition.</summary>
  Similar to the second task here, but in a more difficult setting where you may not manually encode actions.
  Get started by opening [this documentation](https://docs.google.com/document/d/1BxKAFZN1-qfc83GjVYMdsJamU01sngn2LlreuvdxWu0/edit?usp=sharing).
</details>
