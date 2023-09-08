from typing import Any, Dict
import numpy as np
from PIL import Image


################################################################################################
#                                        Target config                                         #
################################################################################################
# features=tfds.features.FeaturesDict({
#     'steps': tfds.features.Dataset({
#         'observation': tfds.features.FeaturesDict({
#             'image': tfds.features.Image(
#                 shape=(128, 128, 3),
#                 dtype=np.uint8,
#                 encoding_format='jpeg',
#                 doc='Main camera RGB observation.',
#             ),
#         }),
#         'action': tfds.features.Tensor(
#             shape=(8,),
#             dtype=np.float32,
#             doc='Robot action, consists of [3x EEF position, '
#                 '3x EEF orientation yaw/pitch/roll, 1x gripper open/close position, '
#                 '1x terminate episode].',
#         ),
#         'discount': tfds.features.Scalar(
#             dtype=np.float32,
#             doc='Discount if provided, default to 1.'
#         ),
#         'reward': tfds.features.Scalar(
#             dtype=np.float32,
#             doc='Reward if provided, 1 on final step for demos.'
#         ),
#         'is_first': tfds.features.Scalar(
#             dtype=np.bool_,
#             doc='True on first step of the episode.'
#         ),
#         'is_last': tfds.features.Scalar(
#             dtype=np.bool_,
#             doc='True on last step of the episode.'
#         ),
#         'is_terminal': tfds.features.Scalar(
#             dtype=np.bool_,
#             doc='True on last step of the episode if it is a terminal step, True for demos.'
#         ),
#         'language_instruction': tfds.features.Text(
#             doc='Language Instruction.'
#         ),
#         'language_embedding': tfds.features.Tensor(
#             shape=(512,),
#             dtype=np.float32,
#             doc='Kona language embedding. '
#                 'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
#         ),
#     })
################################################################################################
#                                                                                              #
################################################################################################

def quaternion_to_euler(q):
    # Extract quaternion components
    w, x, y, z = q
    # Calculate roll (x-axis rotation)
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
    # Calculate pitch (y-axis rotation)
    pitch = np.arcsin(2 * (w * y - z * x))
    # Calculate yaw (z-axis rotation)
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    return np.float32(np.array([roll, pitch, yaw]))

def transform_step(step: Dict[str, Any]) -> Dict[str, Any]:
    """Maps step from source dataset to target dataset config.
       Input is dict of numpy arrays."""
    img = Image.fromarray(step['observation']['image']).resize(
        (128, 128), Image.Resampling.LANCZOS)
    step_eef_angles = quaternion_to_euler(step['observation']['end_effector_state'][3:7])
    transformed_step = {
        'observation': {
            'image': np.array(img),
        },
        'action': np.concatenate(
            [step['observation']['end_effector_state'][:3], step_eef_angles, step['observation']['state'][6:7], np.array([np.float32(step['is_terminal'])])]),
    }

    # copy over all other fields unchanged
    for copy_key in ['discount', 'reward', 'is_first', 'is_last', 'is_terminal',
                     'language_instruction', 'language_embedding']:
        transformed_step[copy_key] = step[copy_key]

    return transformed_step

