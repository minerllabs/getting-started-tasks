# Import necessary libraries
import pprint

# We use opencv/cv2 to draw things and get keypresses
import cv2
import gym
import numpy as np
import minerl


# Keymapping from keys to MineRL actions.
# All actions are zeros by default ("no-op"),
# and they will be updated with keys presented here.
# NOTE: Only one button can be down at a time
KEYBINDINGS = {
    "w": {"forward": 1},
    "s": {"backward": 1},
    "a": {"left": 1},
    "d": {"right": 1},
    # Arrow keys are not registered by cv2.waitKey
    "u": {"camera": [-5.0, 0.0]},
    "j": {"camera": [5.0, 0.0]},
    "h": {"camera": [0.0, -5.0]},
    "k": {"camera": [0.0, 5.0]},

    " ": {"jump": 1},
    "b": {"attack": 1},

    "1": {"craft": "planks"},
    "2": {"craft": "stick"},
    "3": {"craft": "crafting_table"},
    "4": {"nearbyCraft": "wooden_axe"},

    "c": {"place": "crafting_table"},
    "e": {"equip": "wooden_axe"}
}

DEFAULT_BUTTON_INFO = """
INSTRUCTIONS:

Control player by focusing on the image window and using controls below.
The command line window will show the current inventory.

Default buttons
---------------

   W        U
 A-S-D    H-J-K
Movement  Camera

Space = Jump
[b] = Attack

[1] = Craft planks
[2] = Craft sticks
[3] = Craft a crafting table
[4] = Craft a wooden axe (while nearby crafting table)
[c] = Place crafting table
[e] = Equip wooden axe (if any)
---------------
"""


def show_observation_and_get_action(env, observation):
    """Show observation to human observer and wait for a keypress"""
    # Print out the current inventory and item in hand
    # For more documentation, see https://minerl.readthedocs.io/en/latest/environments/index.html#minerlobtaindiamond-v0
    main_hand_item = observation["equipped_items"]["mainhand"]
    inventory = observation["inventory"]
    # Clean up inventory info a bit: if no item in inventory, remove its print
    for key in list(inventory.keys()):
        if inventory[key] == 0:
            _ = inventory.pop(key)
        else:
            # Remove numpy array and just replace with int
            inventory[key] = inventory[key].item()
    print("Main hand item: {}".format(main_hand_item["type"]))
    print("Inventory:\n", pprint.pformat(inventory, indent=4, width=10))

    # Show image
    image = observation["pov"]
    # Make it larger for easier reading
    image = cv2.resize(image, (256, 256))

    # Flip color channels (because OpenCV/cv2 wants BGR instead of RGB)
    cv2.imshow("image", image[..., ::-1])
    # Now wait for a keypress for the action
    key_id = cv2.waitKey(0)
    # key_id is an ASCI code of the pressed character
    key_char = chr(key_id)

    # Create empty action
    action = env.action_space.noop()
    # If pressed key matches one in keybindings,
    # update action.
    if key_char in KEYBINDINGS:
        action.update(KEYBINDINGS[key_char])

    return action


def main():
    """Entry point to the code. Start here"""

    print(DEFAULT_BUTTON_INFO)
    print("Now launching MineRL... This will take a while.")

    # Create MineRL environment
    # This version has basic crafting options available
    env = gym.make("MineRLObtainDiamondDense-v0")
    while True:
        # Play until user says otherwise
        # Reset environment (create a new world)
        # NOTE: This will take a long time, especially on the first time running MineRL!
        observation = env.reset()
        done = False
        while not done:
            # Play until player is dead
            # Ask human player for an action
            action = show_observation_and_get_action(env, observation)
            # Execute action on the environment and proceed to next frame
            observation, reward, done, info = env.step(action)

    # Close environment properly
    env.close()


# Curious why this is needed?
# See https://stackoverflow.com/questions/419163/what-does-if-name-main-do
if __name__ == "__main__":
    main()
