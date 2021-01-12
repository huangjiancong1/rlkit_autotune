import numpy as np
from gym.spaces import Dict, Discrete
import random

from rlkit.data_management.replay_buffer import ReplayBuffer
import pandas as pd


class ObsDictRelabelingBuffer(ReplayBuffer):
    """
    Replay buffer for environments whose observations are dictionaries, such as
        - OpenAI Gym GoalEnv environments. https://blog.openai.com/ingredients-for-robotics-research/
        - multiworld MultitaskEnv. https://github.com/vitchyr/multiworld/

    Implementation details:
     - Only add_path is implemented.
     - Image observations are presumed to start with the 'image_' prefix
     - Every sample from [0, self._size] will be valid.
     - Observation and next observation are saved separately. It's a memory
       inefficient to save the observations twice, but it makes the code
       *much* easier since you no longer have to worry about termination
       conditions.
    """

    def __init__(
            self,
            # automatic_policy_schedule,
            max_size,
            env,
            fraction_goals_rollout_goals=1.0,
            fraction_goals_env_goals=0.0,
            internal_keys=None,
            goal_keys=None,
            observation_key='observation',
            desired_goal_key='desired_goal',
            achieved_goal_key='achieved_goal',
    ):
        if internal_keys is None:
            internal_keys = []
        self.internal_keys = internal_keys
        if goal_keys is None:
            goal_keys = []
        if desired_goal_key not in goal_keys:
            goal_keys.append(desired_goal_key)
        self.goal_keys = goal_keys
        assert isinstance(env.observation_space, Dict)
        assert 0 <= fraction_goals_rollout_goals
        assert 0 <= fraction_goals_env_goals
        assert 0 <= fraction_goals_rollout_goals + fraction_goals_env_goals
        assert fraction_goals_rollout_goals + fraction_goals_env_goals <= 1
        self.max_size = max_size
        self.original_max_size = max_size
        self.env = env
        self.fraction_goals_rollout_goals = fraction_goals_rollout_goals
        self.fraction_goals_env_goals = fraction_goals_env_goals
        self.ob_keys_to_save = [
            observation_key,
            desired_goal_key,
            achieved_goal_key,
        ]
        self.observation_key = observation_key
        self.desired_goal_key = desired_goal_key
        self.achieved_goal_key = achieved_goal_key
        if isinstance(self.env.action_space, Discrete):
            self._action_dim = env.action_space.n
        else:
            self._action_dim = env.action_space.low.size

        self.all_path_lengths = np.zeros((max_size, 1))
        self._intrinsic_rewards = np.zeros((max_size, 1))
        self._extrinsic_rewards = np.zeros((max_size, 1))

        self._actions = np.zeros((max_size, self._action_dim))
        # self._terminals[i] = a terminal was received at time i
        self._terminals = np.zeros((max_size, 1), dtype='uint8')
        # self._obs[key][i] is the value of observation[key] at time i
        self._obs = {}
        self._next_obs = {}
        self.ob_spaces = self.env.observation_space.spaces
        for key in self.ob_keys_to_save + internal_keys:
            assert key in self.ob_spaces, \
                "Key not found in the observation space: %s" % key
            type = np.float64
            if key.startswith('image'):
                type = np.uint8
            self._obs[key] = np.zeros(
                (max_size, self.ob_spaces[key].low.size), dtype=type)
            self._next_obs[key] = np.zeros(
                (max_size, self.ob_spaces[key].low.size), dtype=type)

        self._top = 0 # _top is the first number of the slice'number (index num) ready to replay with the novel transitions
        self._size = 0
        self._actual_size = 0 # TODO: improve from without use

        # Let j be any index in self._idx_to_future_obs_idx[i]
        # Then self._next_obs[j] is a valid next observation for observation i
        self._idx_to_future_obs_idx = [None] * max_size
        self.stable_number_of_elbo = 0

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, **kwargs):
        raise NotImplementedError("Only use add_path")

    def terminate_episode(self):
        pass

    def num_steps_can_sample(self):
        return self.max_size

    def add_path(self, path):
        obs = path["observations"]
        actions = path["actions"]
        rewards = path["rewards"]
        intrinsic_rewards = path["intrinsic_rewards"]
        next_obs = path["next_observations"]
        terminals = path["terminals"]
        path_len = len(rewards)
        actions = flatten_n(actions)
        if isinstance(self.env.action_space, Discrete):
            actions = np.eye(self._action_dim)[actions].reshape((-1, self._action_dim))
        obs = flatten_dict(obs, self.ob_keys_to_save + self.internal_keys)
        next_obs = flatten_dict(next_obs, self.ob_keys_to_save + self.internal_keys)
        obs = preprocess_obs_dict(obs)
        next_obs = preprocess_obs_dict(next_obs)

        #TODO: autotune the replay buffer size 
        try:
            progress_recorder = pd.read_csv(self.automatic_policy_schedule['vae_pkl_path']+'/progress.csv').to_dict('l')
            if len(progress_recorder['vae_trainer/test/loss']) == 1:
                minus_elbo = 0
                old_minus_elbo = minus_elbo
            else:
                minus_elbo = progress_recorder['vae_trainer/test/loss'][-1]
                old_minus_elbo = progress_recorder['vae_trainer/test/loss'][-2]
        except:
            minus_elbo = 0
            old_minus_elbo = minus_elbo

        delta = abs(minus_elbo-old_minus_elbo)
        self.delta_elbo_all_value = delta

        if delta <= self.automatic_policy_schedule['auto_start_threshold'] and minus_elbo > 0:
            self.old_stable_number_of_elbo = self.stable_number_of_elbo
            self.stable_number_of_elbo = minus_elbo
            self.delta_elbo_stable_value = delta
            self.tune_r_size = True
        
        else:
            self.tune_r_size = False

        ## Autotune
        xi = self.automatic_policy_schedule['autotune_xi']
        if self.automatic_policy_schedule['use_autotune']:
            try:
                if len(progress_recorder['vae_trainer/test/loss']) > 1:
                    if self.automatic_policy_schedule['autotune_r_size']:
                        if self.automatic_policy_schedule['autotune_r_size_mode'] == 'max_path_lenth_times_elbo':
                            computed_max_size = int(self.max_path_length*xi*minus_elbo)
                            computed_max_size = np.clip(computed_max_size, 100, 300000) # Fixed to current comupter's capability

                            ######## Compare with the computed_max_size To cut or add length of replay buffer ########

                            # We can not remove the stored transition because the possible future goals may come from the deletion
                            if self.max_size > computed_max_size and self._size <= computed_max_size: 
                                should_delete = self.max_size - computed_max_size
                                self._actions = np.delete(self._actions, np.s_[computed_max_size:self.max_size], axis=0)
                                self.all_path_lengths = np.delete(self.all_path_lengths, np.s_[computed_max_size:self.max_size], axis=0)
                                self._intrinsic_rewards = np.delete(self._intrinsic_rewards, np.s_[computed_max_size:self.max_size], axis=0)
                                self._terminals = np.delete(self._terminals, np.s_[computed_max_size:self.max_size], axis=0)
                                for key in self.ob_keys_to_save + self.internal_keys:
                                    self._obs[key] = np.delete(self._obs[key], np.s_[computed_max_size:self.max_size], axis=0)
                                    self._next_obs[key] = np.delete(self._next_obs[key], np.s_[computed_max_size:self.max_size], axis=0)
                                del self._idx_to_future_obs_idx[computed_max_size:self.max_size]
                                # gc.collect()
                                self.max_size = computed_max_size
                                                                                    
                            ## We can remove the whole trajectory because not possible future goals may come from the deletion if use 'self._top = 0'
                            if self.max_size > computed_max_size and self._size > computed_max_size: 
                                last_traj_idx = self.all_path_lengths[computed_max_size][-1] ## last trajectory idx
                                should_delete = self.max_size - last_traj_idx
                                self._actions = np.delete(self._actions, np.s_[last_traj_idx:self.max_size], axis=0)
                                self.all_path_lengths = np.delete(self.all_path_lengths, np.s_[last_traj_idx:self.max_size], axis=0)
                                self._intrinsic_rewards = np.delete(self._intrinsic_rewards, np.s_[last_traj_idx:self.max_size], axis=0)
                                self._terminals = np.delete(self._terminals, np.s_[last_traj_idx:self.max_size], axis=0)
                                for key in self.ob_keys_to_save + self.internal_keys:
                                    self._obs[key] = np.delete(self._obs[key], np.s_[last_traj_idx:self.max_size], axis=0)
                                    self._next_obs[key] = np.delete(self._next_obs[key], np.s_[last_traj_idx:self.max_size], axis=0)
                                del self._idx_to_future_obs_idx[last_traj_idx:self.max_size]
                                # gc.collect()
                                self.max_size = last_traj_idx
                            
                            if self._size > computed_max_size:
                                ## Count how many useful state from buffer and the useful deletion
                                self._actual_size = self._size
                                # self.max_size = self._size ##TODO: why should'n use here currently?

                            ## To add the transitions grid to save the future samples from self.max_size
                            if self.max_size < computed_max_size and self._actual_size <= self.max_size:
                                should_add = computed_max_size - self.max_size
                                self._actions = np.pad(self._actions,((0,should_add),(0,0)),'constant',constant_values = 0)
                                self.all_path_lengths = np.pad(self.all_path_lengths,((0,should_add),(0,0)),'constant',constant_values = 0)
                                self._intrinsic_rewards = np.pad(self._intrinsic_rewards,((0,should_add),(0,0)),'constant',constant_values = 0)
                                self._terminals = np.pad(self._terminals,((0,should_add),(0,0)),'constant',constant_values = 0)
                                for key in self.ob_keys_to_save + self.internal_keys:
                                    self._obs[key] = np.pad(self._obs[key], ((0,should_add),(0,0)), 'constant', constant_values = 0)
                                    self._next_obs[key] = np.pad(self._next_obs[key], ((0,should_add),(0,0)), 'constant', constant_values = 0)
                                for i in range(should_add):
                                    self._idx_to_future_obs_idx.append(None)
                                self.max_size = computed_max_size
                            
                            ## To add the transitions grid to save the future samples from self._actual_size
                            if self.max_size < computed_max_size and computed_max_size >= self._actual_size: 
                                should_add = computed_max_size - self._actual_size
                                self._actions = np.pad(self._actions,((0,should_add),(0,0)),'constant',constant_values = 0)
                                self.all_path_lengths = np.pad(self.all_path_lengths,((0,should_add),(0,0)),'constant',constant_values = 0)
                                self._intrinsic_rewards = np.pad(self._intrinsic_rewards,((0,should_add),(0,0)),'constant',constant_values = 0)
                                self._terminals = np.pad(self._terminals,((0,should_add),(0,0)),'constant',constant_values = 0)
                                for key in self.ob_keys_to_save + self.internal_keys:
                                    self._obs[key] = np.pad(self._obs[key], ((0,should_add),(0,0)), 'constant', constant_values = 0)
                                    self._next_obs[key] = np.pad(self._next_obs[key], ((0,should_add),(0,0)), 'constant', constant_values = 0)
                                for i in range(should_add):
                                    self._idx_to_future_obs_idx.append(None)
                                self.max_size = computed_max_size

                            ## If equal of when equal after previous loop's computed
                            if self.max_size == computed_max_size:
                                self.max_size = computed_max_size
                    else: 
                        self.max_size = self.original_max_size

            except:
                pass

        if self._top + path_len >= self.max_size:
            """
            All of this logic is to handle wrapping the pointer when the
            replay buffer gets full.
            """
            self._top = 0
            self._size = self.max_size
            slc = np.s_[self._top:self._top + path_len, :]

            self._actions[slc] = actions
            self.all_path_lengths[slc] = path_len
            self._intrinsic_rewards[slc] = intrinsic_rewards
            self._terminals[slc] = terminals
            
            for key in self.ob_keys_to_save + self.internal_keys:
                self._obs[key][slc] = obs[key]
                self._next_obs[key][slc] = next_obs[key]
            for i in range(self._top, self._top + path_len):
                self._idx_to_future_obs_idx[i] = np.arange( 
                    i, self._top + path_len
                )
                
        else:
            slc = np.s_[self._top:self._top + path_len, :]

            self._actions[slc] = actions
            self.all_path_lengths[slc] = path_len
            self._intrinsic_rewards[slc] = intrinsic_rewards
            self._terminals[slc] = terminals
            
            for key in self.ob_keys_to_save + self.internal_keys:
                self._obs[key][slc] = obs[key]
                self._next_obs[key][slc] = next_obs[key]
            for i in range(self._top, self._top + path_len):
                self._idx_to_future_obs_idx[i] = np.arange(
                    i, self._top + path_len
                )
        self._top = (self._top + path_len) % self.max_size
        self._size = min(self._size + path_len, self.max_size)  

    def _sample_indices(self, batch_size):
        actual_length = self.count_the_not_none(self._idx_to_future_obs_idx)
        return np.random.randint(0, actual_length-1, batch_size)

    def count_the_not_none(self, future_idx):
        num_of_none=1
        for i in future_idx:
            if i is None:
                num_of_none+=1
        return len(future_idx)-num_of_none

    def _sample_odd_indices(self, batch_size):
        random_num = np.random.randint(0, int(self._size/2), batch_size)
        return random_num*2+1
      
    def _sample_even_indices(self, batch_size):
        random_num = np.random.randint(0, int(self._size/2), batch_size)
        return random_num*2

    def replay_buffer_stores(self,):
        # How many trsnditiond stored in replay buffer
        return self._size

    def replay_buffer_stores_actual(self,):
        # How many trsnditiond stored in replay buffer
        return self._actual_size

    def return_size_of_replay_buffer(self,):
        size_of_replay_buffer = self.max_size
        return size_of_replay_buffer
        
    def random_batch(self, batch_size):
        indices = self._sample_indices(batch_size)
        resampled_goals = self._next_obs[self.desired_goal_key][indices]

        num_env_goals = int(batch_size * self.fraction_goals_env_goals)
        num_rollout_goals = int(batch_size * self.fraction_goals_rollout_goals)
        num_future_goals = batch_size - (num_env_goals + num_rollout_goals)
        new_obs_dict = self._batch_obs_dict(indices)
        new_next_obs_dict = self._batch_next_obs_dict(indices)

        if num_env_goals > 0:
            env_goals = self.env.sample_goals(num_env_goals)
            env_goals = preprocess_obs_dict(env_goals)
            last_env_goal_idx = num_rollout_goals + num_env_goals
            resampled_goals[num_rollout_goals:last_env_goal_idx] = (
                env_goals[self.desired_goal_key]
            )
            for goal_key in self.goal_keys:
                new_obs_dict[goal_key][num_rollout_goals:last_env_goal_idx] = \
                    env_goals[goal_key]
                new_next_obs_dict[goal_key][
                num_rollout_goals:last_env_goal_idx] = \
                    env_goals[goal_key]
        if num_future_goals > 0:
            ## -----------1d469a509b797ca04a39b8734c1816ca7d108fc8--------
            future_obs_idxs = []
            for i in indices[-num_future_goals:]:
                possible_future_obs_idxs = self._idx_to_future_obs_idx[i]
                # This is generally faster than random.choice. Makes you wonder what
                # random.choice is doing
                num_options = len(possible_future_obs_idxs)
                next_obs_i = int(np.random.randint(0, num_options))
                future_obs_idxs.append(possible_future_obs_idxs[next_obs_i])
            future_obs_idxs = np.array(future_obs_idxs)
            ## ---------------------------------------------------------

            ## ---------5274672e9ff6481def0ffed61cd1b1c52210a840----------
            # future_indices = indices[-num_future_goals:]
            # possible_future_obs_lens = np.array([
            #     len(self._idx_to_future_obs_idx[i]) for i in future_indices
            # ])
            # # Faster than a naive for-loop.
            # # See https://github.com/vitchyr/rlkit/pull/112 for details.
            # next_obs_idxs = (
            #     np.random.random(num_future_goals) * possible_future_obs_lens
            # ).astype(np.int)
            # future_obs_idxs = np.array([
            #     self._idx_to_future_obs_idx[ids][next_obs_idxs[i]]
            #     for i, ids in enumerate(future_indices)
            # ])
            ## ---------------------------------------------------------            
            resampled_goals[-num_future_goals:] = self._next_obs[
                self.achieved_goal_key
            ][future_obs_idxs]

            for goal_key in self.goal_keys:
                new_obs_dict[goal_key][-num_future_goals:] = \
                    self._next_obs[goal_key][future_obs_idxs]
                new_next_obs_dict[goal_key][-num_future_goals:] = \
                    self._next_obs[goal_key][future_obs_idxs]

        new_obs_dict[self.desired_goal_key] = resampled_goals
        new_next_obs_dict[self.desired_goal_key] = resampled_goals
        new_obs_dict = postprocess_obs_dict(new_obs_dict)
        new_next_obs_dict = postprocess_obs_dict(new_next_obs_dict)
        # resampled_goals must be postprocessed as well
        resampled_goals = new_next_obs_dict[self.desired_goal_key]

        new_actions = self._actions[indices]
        new_intrinsic_rewards = self._intrinsic_rewards[indices]
        """
        For example, the environments in this repo have batch-wise
        implementations of computing rewards:

        https://github.com/vitchyr/multiworld
        """
        if hasattr(self.env, 'compute_rewards'):
            new_rewards = self.env.compute_rewards(
                new_actions,
                new_next_obs_dict,
            )
        else:  # Assuming it's a (possibly wrapped) gym GoalEnv
            new_rewards = np.ones((batch_size, 1))
            for i in range(batch_size):
                new_rewards[i] = self.env.compute_reward(
                    new_next_obs_dict[self.achieved_goal_key][i],
                    new_next_obs_dict[self.desired_goal_key][i],
                    None
                )
        extrinsic_rewards = new_rewards.reshape(-1, 1)
        new_rewards = new_rewards.reshape(-1, 1)+new_intrinsic_rewards

        new_obs = new_obs_dict[self.observation_key]
        new_next_obs = new_next_obs_dict[self.observation_key]
        batch = {
            'observations': new_obs,
            'actions': new_actions,
            'rewards': new_rewards,
            'terminals': self._terminals[indices],
            'next_observations': new_next_obs,
            'resampled_goals': resampled_goals,
            'indices': np.array(indices).reshape(-1, 1),
            'extrinsic_rewards': extrinsic_rewards,
        }
        return batch

    def _batch_obs_dict(self, indices):
        return {
            key: self._obs[key][indices]
            for key in self.ob_keys_to_save
        }

    def _batch_next_obs_dict(self, indices):
        return {
            key: self._next_obs[key][indices]
            for key in self.ob_keys_to_save
        }

def flatten_n(xs):
    xs = np.asarray(xs)
    return xs.reshape((xs.shape[0], -1))


def flatten_dict(dicts, keys):
    """
    Turns list of dicts into dict of np arrays
    """
    return {
        key: flatten_n([d[key] for d in dicts])
        for key in keys
    }

def preprocess_obs_dict(obs_dict):
    """
    Apply internal replay buffer representation changes: save images as bytes
    """
    for obs_key, obs in obs_dict.items():
        if 'image' in obs_key and obs is not None:
            obs_dict[obs_key] = unnormalize_image(obs)
    return obs_dict

def postprocess_obs_dict(obs_dict):
    """
    Undo internal replay buffer representation changes: save images as bytes
    """
    for obs_key, obs in obs_dict.items():
        if 'image' in obs_key and obs is not None:
            obs_dict[obs_key] = normalize_image(obs)
    return obs_dict

def normalize_image(image):
    assert image.dtype == np.uint8
    return np.float64(image) / 255.0


def unnormalize_image(image):
    assert image.dtype != np.uint8
    return np.uint8(image * 255.0)
