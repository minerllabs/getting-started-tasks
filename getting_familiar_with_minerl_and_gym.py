from time import sleep

import cv2
import gym
import minerl


SLEEP_TIME = 0.1

"""
The “step-loop” is common way do to agent-environment interaction. Implement this missing loop in this code. See this for documentation: https://gym.openai.com/
First, use environment "CartPole-v0" to use it to see how random agent plays the game.
Get familiar with the environment CartPole-v0. Important things you need to figure out:
    What is the goal?
    What is the state information (what does the state from env.step(action) mean)?
    What are actions (what does the action in env.step(action) do)? What are the different actions?
    What is reward?
    What is the starting state? Does it change between different episodes (games)?
    Is environment deterministic or stochastic? Environment is deterministic if you can perfectly predict what happens with every action in every state.
Now that you know how CartPole-v0 works, change to "MineRLObtainDiamond-v0", repeat the above steps by changing the environment in `gym.make`
"""


def render(observation, environment):
    """A function for rendering MineRL environments. You do not need to worry about this function"""
    if isinstance(environment.unwrapped, minerl.env._singleagent._SingleAgentEnv):
        # Environment is a MineRL one, use OpenCV image showing to show image
        # Make it larger for easier reading
        image = observation["pov"]
        image = cv2.resize(image, (256, 256))
        cv2.imshow("minerl-image", image[..., ::-1])
        # Refresh image
        _ = cv2.waitKey(1)
    else:
        # Regular render
        environment.render()


def main():
    # Create environment
    # Start by using CartPole-v0 for fast debugging and running!
    # Once code works, change to "MineRLObtainDiamond-v0"
    environment = gym.make("CartPole-v0")
    for episode_counter in range(3):
        # Play three episodes

        # Reset the environment (NOTE: With MineRL, this will take time, but with Cartpole is fast)
        observation = environment.reset()

        # Show the game situation for us to see what is going on
        # (Note: Normally you would use `environment.render()`, but because of MineRL
        # we have a different setup)
        render(observation, environment)
        # Wait a moment to give slow brains some time to process the information
        sleep(SLEEP_TIME)

        raise NotImplementedError("Implement 'step-loop' here to play one episode")

        # TODO step-loop
        # A while-loop until game returns done == True:
        #   - Pick a random action: `environment.action_space` tells what kind actions are available.
        #       You can use `environment.action_space.sample()` to get a random action
        #   - Step game with `environment.step(action)`. See documentation: http://gym.openai.com/docs/
        #   - Print what kind of values you got from the game (observation, reward, done)
        #   - Render current state of the game (see the use of `render` function few lines above)
        #   - Sleep a little bit with `sleep(SLEEP_TIME)` to slow down game 

    # Close environment properly
    environment.close()


# Curious why this is needed?
# See https://stackoverflow.com/questions/419163/what-does-if-name-main-do
if __name__ == "__main__":
    main()
