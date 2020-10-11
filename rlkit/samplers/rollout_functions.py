import numpy as np
import torch


def multitask_rollout(
        env,
        agent,
        other_variant,
        max_path_length=np.inf,
        render=False,
        render_kwargs=None,
        observation_key=None,
        desired_goal_key=None,
        get_action_kwargs=None,
        return_dict_obs=False,
):
    if render_kwargs is None:
        render_kwargs = {}
    if get_action_kwargs is None:
        get_action_kwargs = {}
    dict_obs = []
    dict_next_obs = []
    observations = []
    actions = []
    rewards = []
    intrinsic_rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    next_observations = []
    path_length = 0
    agent.reset()
    o = env.reset()
    if render:
        env.render(**render_kwargs)
    goal = o[desired_goal_key]
    while path_length < max_path_length:
        dict_obs.append(o)
        if observation_key:
            o = o[observation_key]
        new_obs = np.hstack((o, goal))
        a, agent_info = agent.get_action(new_obs, **get_action_kwargs)
        next_o, r, d, env_info = env.step(a)

        if other_variant['use_intrinsic_bonus']:
            # use vae model methods
            # current_vae = torch.load(via_variance['vae_pkl_path']+'/online_vae.pkl') # load vae modelfrom the file which saved
            current_vae = env.vae

            # calculate current state's log_prob
            image_o = next_o['image_observation']
            input_data = torch.Tensor(image_o.tolist()).cuda().view(1, len(image_o))
            reconstructions, obs_distribution_params, latent_distribution_params = current_vae(input_data)
            log_prob = current_vae.logprob(input_data, obs_distribution_params).cpu().detach().numpy()
            kl = current_vae.kl_divergence(latent_distribution_params).cpu().detach().numpy()

            # read the csv recorder for the analysis
            try:
                progress_recorder = pd.read_csv(other_variant['vae_pkl_path']+'/progress.csv').to_dict('l')
                if len(progress_recorder['vae_trainer/test/KL']) == 1:
                    kl_old = 0
                    log_prob_old = 0
                else:
                    kl_old = progress_recorder['vae_trainer/test/KL'][-1]
                    log_prob_old = progress_recorder['vae_trainer/test/log prob'][-1]
            except:
                kl_old = 0
                log_prob_old = 0

            ## coefficient method like large-scaler paper mentions
            # coefficient_all=0.0005
            coefficient_all=0.05 


            # Different Intrinsic Rewards Components
            ## enlarge H(S)
            if other_variant['intrinsic_reward'] == 'minus_log':
                r_i = coefficient_all*(-log_prob + log_prob_old)  # intrinsic reward

            
        else:
            log_prob = 0
            kl = 0
            kl_old = 0
            
            r_i = 0

        if render:
            env.render(**render_kwargs)
        observations.append(o)
        rewards.append(r)
        intrinsic_rewards.append(r_i)  # intrinsic reward      
        terminals.append(d)
        actions.append(a)
        next_observations.append(next_o)
        dict_next_obs.append(next_o)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    next_observations = np.array(next_observations)
    if return_dict_obs:
        observations = dict_obs
        next_observations = dict_next_obs
    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        extrinsic_rewards=np.array(rewards).reshape(-1, 1),
        intrinsic_rewards=np.array(intrinsic_rewards).reshape(-1, 1), # intrinsic reward
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
        goals=np.repeat(goal[None], path_length, 0),
        full_observations=dict_obs,
    )


def rollout(
        env,
        agent,
        max_path_length=np.inf,
        render=False,
        render_kwargs=None,
):
    """
    The following value for the following keys will be a 2D array, with the
    first dimension corresponding to the time dimension.
     - observations
     - actions
     - rewards
     - next_observations
     - terminals

    The next two elements will be lists of dictionaries, with the index into
    the list being the index into the time
     - agent_infos
     - env_infos
    """
    if render_kwargs is None:
        render_kwargs = {}
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    agent.reset()
    next_o = None
    path_length = 0
    if render:
        env.render(**render_kwargs)
    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        next_o, r, d, env_info = env.step(a)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
        if render:
            env.render(**render_kwargs)

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
        next_o = np.array([next_o])
    next_observations = np.vstack(
        (
            observations[1:, :],
            np.expand_dims(next_o, 0)
        )
    )
    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
    )
