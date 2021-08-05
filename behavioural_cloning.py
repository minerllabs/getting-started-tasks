from tqdm import tqdm
import numpy as np
import torch as th
from torch import nn
import gym
import minerl


"""
Your task: Implement behavioural cloning for MineRLTreechop-v0.

Behavioural cloning is perhaps the simplest way of using a dataset of demonstrations to train an agent:
learn to predict what actions they would take, and take those actions.
In other machine learning terms, this is almost like building a classifier to classify observations to
different actions, and taking those actions.

For simplicity, we build a limited set of actions ("agent actions"), map dataset actions to these actions
and train on the agent actions. During evaluation, we transform these agent actions (integerse) back into
MineRL actions (dictionaries).

To do this task, fill in the "TODO"s and remove `raise NotImplementedError`s.

Note: For this task you need to download the "MineRLTreechop-v0" dataset. See here:
https://minerl.readthedocs.io/en/latest/tutorials/data_sampling.html#downloading-the-minerl-dataset-with-minerl-data-download
"""


class ConvNet(nn.Module):
    """
    :param input_shape: A three-item tuple telling image dimensions in (C, H, W)
    :param output_dim: Dimensionality of the output vector
    """

    def __init__(self, input_shape, output_dim):
        super().__init__()
        # TODO Create a torch neural network here to turn images (of shape `input_shape`) into
        #      a vector of shape `output_dim`. This output_dim matches number of available actions.
        #      See examples of doing CNN networks here https://pytorch.org/tutorials/beginner/nn_tutorial.html#switch-to-cnn
        raise NotImplementedError("TODO implement a simple convolutional neural network here")

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # TODO with the layers you created in __init__, transform the `observations` (a tensor of shape (B, C, H, W)) to
        #      a tensor of shape (B, D), where D is the `output_dim`
        raise NotImplementedError("TODO implement forward function of the neural network")


def agent_action_to_environment(noop_action, agent_action):
    """
    Turn an agent action (an integer) into an environment action.
    This should match `environment_action_batch_to_agent_actions`,
    e.g. if attack=1 action was mapped to agent_action=0, then agent_action=0
    should be mapped back to attack=1.

    noop_action is a MineRL action that does nothing. You may want to
    use this as a template for the action you return.
    """
    raise NotImplementedError("TODO implement agent_action_to_environment (see docstring)")


def environment_action_batch_to_agent_actions(dataset_actions):
    """
    Turn a batch of actions from environment (from BufferedBatchIterator) to a numpy
    array of agent actions.

    Agent actions _have to_ start from 0 and go up from there!

    For MineRLTreechop, you want to have actions for the following at the very least:
    - Forward movement
    - Jumping
    - Turning camera left, right, up and down
    - Attack

    For example, you could have seven agent actions that mean following:
    0 = forward
    1 = jump
    2 = turn camera left
    3 = turn camera right
    4 = turn camera up
    5 = turn camera down
    6 = attack

    This should match `agent_action_to_environment`, by converting dictionary
    actions into individual integeres.

    If dataset action (dict) does not have a mapping to agent action (int),
    then set it "-1"
    """
    # There are dummy dimensions of shape one
    batch_size = len(dataset_actions["camera"])
    actions = np.zeros((batch_size,), dtype=np.int)

    for i in range(batch_size):
        # TODO this will make all actions invalid. Replace with something
        # more clever
        actions[i] = -1
        raise NotImplementedError("TODO map dataset action at index i to an agent action, or if no mapping, -1")
    return actions


def train():
    # Path to where MineRL dataset resides (should contain "MineRLTreechop-v0" directory)
    DATA_DIR = "."
    # How many times we train over dataset and how large batches we use.
    # Larger batch size takes more memory but generally provides stabler learning.
    EPOCHS = 1
    BATCH_SIZE = 32

    # TODO create data iterators for going over MineRL data using BufferedBatchIterator
    #      https://minerl.readthedocs.io/en/latest/tutorials/data_sampling.html#sampling-the-dataset-with-buffered-batch-iter
    #      NOTE: You have to download the Treechop dataset first for this to work, see:
    #           https://minerl.readthedocs.io/en/latest/tutorials/data_sampling.html#downloading-the-minerl-dataset-with-minerl-data-download
    raise NotImplementedError("TODO create dataset samplers")
    iterator = None

    number_of_actions = None
    # TODO we need to tell the network how many possible actions there are,
    #      so assign the value in above variable
    raise NotImplementedError("TODO add number of actions to `number_of_actions`")
    network = ConvNet((3, 64, 64), number_of_actions).cuda()
    # TODO create optimizer and loss functions for training
    #      see examples here https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
    raise NotImplementedError("TODO Create an optimizer and a loss function.")
    optimizer = None
    loss_function = None

    iter_count = 0
    losses = []
    for dataset_obs, dataset_actions, _, _, _ in tqdm(iterator.buffered_batch_iter(num_epochs=EPOCHS, batch_size=BATCH_SIZE)):
        # We only use camera observations here
        obs = dataset_obs["pov"].astype(np.float32)
        # Transpose observations to be channel-first (BCHW instead of BHWC)
        obs = obs.transpose(0, 3, 1, 2)
        # Normalize observations, otherwise the neural network will get spooked
        obs /= 255.0

        # Turn dataset actions into agent actions
        actions = environment_action_batch_to_agent_actions(dataset_actions)
        assert actions.shape == (obs.shape[0],), "Array from environment_action_batch_to_agent_actions should be of shape {}".format((obs.shape[0],))

        # Remove samples that had no corresponding action
        mask = actions != -1
        obs = obs[mask]
        actions = actions[mask]

        # TODO perform optimization step:
        # - Predict actions using the neural network (input is `obs`)
        # - Compute loss with the predictions and true actions. Store loss into variable `loss`
        # - Use optimizer to do a single update step
        # See https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html 
        # for a tutorial
        # NOTE: Variables `obs` and `actions` are numpy arrays. You need to convert them into torch tensors.

        # Keep track of how training is going by printing out the loss
        iter_count += 1
        losses.append(loss.item())
        if (iter_count % 1000) == 0:
            mean_loss = sum(losses) / len(losses)
            tqdm.write("Iteration {}. Loss {:<10.3f}".format(iter_count, mean_loss))
            losses.clear()

    # Store the network
    th.save(network, "behavioural_cloning.pth")


def enjoy():
    # Load up the trained network
    network = th.load("behavioural_cloning.pth").cuda()

    env = gym.make('MineRLTreechop-v0')

    # Play 10 games with the model
    for game_i in range(10):
        obs = env.reset()
        done = False
        reward_sum = 0
        while not done:
            # TODO Process the observation:
            #   - Take only the camera observation
            #   - Add/remove batch dimensions
            #   - Transpose image (needs to be channels-last)
            #   - Normalize image
            #   - Store network output to `logits`
            # For hints, see what preprocessing was done during training
            raise NotImplementedError("TODO process the observation and run it through network")
            logits = None
            # Turn logits into probabilities
            probabilities = th.softmax(logits, dim=1)[0]
            # Into numpy
            probabilities = probabilities.detach().cpu().numpy()
            # TODO Pick an action based from the probabilities above.
            # The `probabilities` vector tells the probability of choosing one of the agent actions.
            # You have two options:
            # 1) Pick action with the highest probability
            # 2) Sample action based on probabilities
            # Option 2 works better emperically.
            agent_action = None

            noop_action = env.action_space.noop()
            environment_action = agent_action_to_environment(noop_action, agent_action)

            obs, reward, done, info = env.step(environment_action)
            reward_sum += reward
        print("Game {}, total reward {}".format(game_i, reward_sum))

    env.close()


if __name__ == "__main__":
    # First train the model...
    train()
    # ... then play it on the environment to see how it does
    enjoy()
