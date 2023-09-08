import numpy as np
import tqdm
import os

N_TRAIN_EPISODES = 100
N_VAL_EPISODES = 100

EPISODE_LENGTH = 10

def save_episode(path, main_camera_inage_hist, wrist_camera_image_hist, state_hist, action_hist, language_instruction_hist):
    """ 
        image_hist: list of np arrys of shape (img_s, img_s, 3)
        wrist_image_hist: list of np arrys of shape (img_s, img_s, 3)
        state_hist: list of np arrys of shape (n_dof,)
        action_hist: list of np arrys of shape (n_dof,)
        language_instruction_hist: list of strings
    """
    episode = []
    for step in range(EPISODE_LENGTH):
        episode.append({
            'image': np.asarray(main_camera_inage_hist[step], dtype=np.uint8),
            'wrist_image': np.asarray(wrist_camera_image_hist[step], dtype=np.uint8),
            'state': np.asarray(state_hist[step], dtype=np.float32),
            'action': np.asarray(action_hist[step], dtype=np.float32),
            'language_instruction': language_instruction_hist[step],
        })
    np.save(path, episode)

def create_fake_episode(path):
    episode = []
    for step in range(EPISODE_LENGTH):
        episode.append({
            'image': np.asarray(np.random.rand(64, 64, 3) * 255, dtype=np.uint8),
            'wrist_image': np.asarray(np.random.rand(64, 64, 3) * 255, dtype=np.uint8),
            'state': np.asarray(np.random.rand(10), dtype=np.float32),
            'action': np.asarray(np.random.rand(10), dtype=np.float32),
            'language_instruction': 'dummy instruction',
        })
    np.save(path, episode)


# create fake episodes for train and validation
print("Generating train examples...")
os.makedirs('data/train', exist_ok=True)
for i in tqdm.tqdm(range(N_TRAIN_EPISODES)):
    create_fake_episode(f'data/train/episode_{i}.npy')

print("Generating val examples...")
os.makedirs('data/val', exist_ok=True)
for i in tqdm.tqdm(range(N_VAL_EPISODES)):
    create_fake_episode(f'data/val/episode_{i}.npy')

print('Successfully created example data!')
