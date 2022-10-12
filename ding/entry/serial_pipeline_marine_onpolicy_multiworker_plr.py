from typing import Union, Optional, List, Any, Tuple
import os
import torch
from ditk import logging
from functools import partial
from tensorboardX import SummaryWriter
from copy import deepcopy

from ding.envs import get_vec_env_setting, create_env_manager
from ding.worker import BaseLearner, InteractionSerialEvaluator, BaseSerialCommander, create_buffer, \
    create_serial_collector
from ding.config import read_config, compile_config
from ding.policy import create_policy, PolicyFactory
from ding.reward_model import create_reward_model
from ding.utils import set_pkg_seed
from ding.data.level_replay.level_sampler import LevelSampler
from ding.policy.common_utils import default_preprocess_learn

def generate_seeds(num_seeds=3, base_seed=0):
    return [base_seed + i for i in range(num_seeds)]

def serial_pipeline_marine_onpolicy_singlelearner_plr(
        input_cfg: Union[str, Tuple[dict, dict]],
        seed: int = 0,
        env_setting: Optional[List[Any]] = None,
        model: Optional[torch.nn.Module] = None,
        max_train_iter: Optional[int] = int(1e10),
        max_env_step: Optional[int] = int(1e10),
) -> 'Policy':  # noqa
    """
    Overview:
        Serial pipeline entry on-policy RL.
    Arguments:
        - input_cfg (:obj:`Union[str, Tuple[dict, dict]]`): Config in dict type. \
            ``str`` type means config file path. \
            ``Tuple[dict, dict]`` type means [user_config, create_cfg].
        - seed (:obj:`int`): Random seed.
        - env_setting (:obj:`Optional[List[Any]]`): A list with 3 elements: \
            ``BaseEnv`` subclass, collector env config, and evaluator env config.
        - model (:obj:`Optional[torch.nn.Module]`): Instance of torch.nn.Module.
        - max_train_iter (:obj:`Optional[int]`): Maximum policy update iterations in training.
        - max_env_step (:obj:`Optional[int]`): Maximum collected environment interaction steps.
    Returns:
        - policy (:obj:`Policy`): Converged policy.
    """
    # if isinstance(input_cfg, str):
    #     cfg, create_cfg = read_config(input_cfg)
    # else:
    #     cfg, create_cfg = deepcopy(input_cfg)
    cfg_1011, create_cfg_1011 = deepcopy(input_cfg)
    cfg_1011.exp_name = 'marine_multihead_singlelearner_10m11m_mappo_seed'+str(seed)
    cfg_1011.env.map_name = '10m_vs_11m'
    cfg_89, create_cfg_89 = deepcopy(input_cfg)
    cfg_89.exp_name = 'marine_multihead_singlelearner_8m9m_mappo_seed'+str(seed)
    cfg_89.env.map_name = '8m_vs_9m'
    cfg_56, create_cfg_56 = deepcopy(input_cfg)
    cfg_56.exp_name = 'marine_multihead_singlelearner_5m6m_mappo_seed'+str(seed)
    cfg_56.env.map_name = '5m_vs_6m'
    create_cfg_1011.policy.type = create_cfg_1011.policy.type + '_command'
    create_cfg_89.policy.type = create_cfg_89.policy.type + '_command'
    create_cfg_56.policy.type = create_cfg_56.policy.type + '_command'
    env_fn = None if env_setting is None else env_setting[0]
    cfg_1011 = compile_config(cfg_1011, seed=seed, env=env_fn, auto=True, create_cfg=create_cfg_1011, save_cfg=True)
    cfg_89 = compile_config(cfg_89, seed=seed, env=env_fn, auto=True, create_cfg=create_cfg_89, save_cfg=True)
    cfg_56 = compile_config(cfg_56, seed=seed, env=env_fn, auto=True, create_cfg=create_cfg_56, save_cfg=True)
    # Create main components: env, policy
    # if env_setting is None:
    #     env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
    # else:
    #     env_fn, collector_env_cfg, evaluator_env_cfg = env_setting
    collector_env_num = cfg_1011.env.collector_env_num
    env_fn_1011, collector_env_cfg_1011, evaluator_env_cfg_1011 = get_vec_env_setting(cfg_1011.env)
    env_fn_89, collector_env_cfg_89, evaluator_env_cfg_89 = get_vec_env_setting(cfg_89.env)
    env_fn_56, collector_env_cfg_56, evaluator_env_cfg_56 = get_vec_env_setting(cfg_56.env)
    collector_env_1011 = create_env_manager(cfg_1011.env.manager, [partial(env_fn_1011, cfg=c) for c in collector_env_cfg_1011])
    collector_env_89 = create_env_manager(cfg_89.env.manager, [partial(env_fn_89, cfg=c) for c in collector_env_cfg_89])
    collector_env_56 = create_env_manager(cfg_56.env.manager, [partial(env_fn_56, cfg=c) for c in collector_env_cfg_56])
    evaluator_env_1011 = create_env_manager(cfg_1011.env.manager, [partial(env_fn_1011, cfg=c) for c in evaluator_env_cfg_1011])
    evaluator_env_89 = create_env_manager(cfg_89.env.manager, [partial(env_fn_89, cfg=c) for c in evaluator_env_cfg_89])
    evaluator_env_56 = create_env_manager(cfg_56.env.manager, [partial(env_fn_56, cfg=c) for c in evaluator_env_cfg_56])
    collector_env_1011.seed(cfg_1011.seed)
    collector_env_89.seed(cfg_89.seed)
    collector_env_56.seed(cfg_56.seed)
    evaluator_env_1011.seed(cfg_1011.seed, dynamic_seed=False)
    evaluator_env_89.seed(cfg_89.seed, dynamic_seed=False)
    evaluator_env_56.seed(cfg_56.seed, dynamic_seed=False)

    train_seeds = generate_seeds()
    level_sampler = LevelSampler(
        train_seeds, cfg_1011.policy.model.agent_obs_shape, cfg_1011.policy.model.action_shape, collector_env_num, cfg_1011.level_replay
    )

    set_pkg_seed(cfg_56.seed, use_cuda=cfg_56.policy.cuda)
    policy = create_policy(cfg_56.policy, model=model, enable_field=['learn', 'collect', 'eval', 'command'])

    # Create worker components: learner, collector, evaluator, replay buffer, commander.
    tb_logger_1011 = SummaryWriter(os.path.join('./{}/log/'.format(cfg_1011.exp_name), 'serial'))
    tb_logger_89 = SummaryWriter(os.path.join('./{}/log/'.format(cfg_89.exp_name), 'serial'))
    tb_logger_56 = SummaryWriter(os.path.join('./{}/log/'.format(cfg_56.exp_name), 'serial'))
    # learner_1011 = BaseLearner(cfg_1011.policy.learn.learner, policy.learn_mode, tb_logger_1011, exp_name=cfg_1011.exp_name)
    # learner_89 = BaseLearner(cfg_89.policy.learn.learner, policy.learn_mode, tb_logger_89, exp_name=cfg_89.exp_name)
    # learner_56 = BaseLearner(cfg_56.policy.learn.learner, policy.learn_mode, tb_logger_56, exp_name=cfg_56.exp_name)
    learner = BaseLearner(cfg_1011.policy.learn.learner, policy.learn_mode, tb_logger_1011, exp_name=cfg_1011.exp_name)
    collector_1011 = create_serial_collector(
        cfg_1011.policy.collect.collector,
        env=collector_env_1011,
        policy=policy.collect_mode,
        tb_logger=tb_logger_1011,
        exp_name=cfg_1011.exp_name
    )    
    collector_89 = create_serial_collector(
        cfg_89.policy.collect.collector,
        env=collector_env_89,
        policy=policy.collect_mode,
        tb_logger=tb_logger_89,
        exp_name=cfg_89.exp_name
    )
    collector_56 = create_serial_collector(
        cfg_56.policy.collect.collector,
        env=collector_env_56,
        policy=policy.collect_mode,
        tb_logger=tb_logger_56,
        exp_name=cfg_56.exp_name
    )
    evaluator_1011 = InteractionSerialEvaluator(
        cfg_1011.policy.eval.evaluator, evaluator_env_1011, policy.eval_mode, tb_logger_1011, exp_name=cfg_1011.exp_name
    )
    evaluator_89 = InteractionSerialEvaluator(
        cfg_89.policy.eval.evaluator, evaluator_env_89, policy.eval_mode, tb_logger_89, exp_name=cfg_89.exp_name
    )
    evaluator_56 = InteractionSerialEvaluator(
        cfg_56.policy.eval.evaluator, evaluator_env_56, policy.eval_mode, tb_logger_56, exp_name=cfg_56.exp_name
    )
    # commander is useless in ppo
    commander = BaseSerialCommander(
        cfg_56.policy.other.commander, learner, collector_56, evaluator_56, None, policy.command_mode
    )

    # ==========
    # Main loop
    # ==========
    # Learner's before_run hook.
    learner.call_hook('before_run')

    seed = int(level_sampler.sample('sequential'))

    while True:
        collect_kwargs = commander.step()
        # Evaluate policy performance
        if evaluator_1011.should_eval(learner.train_iter):
            stop_1011, _ = evaluator_1011.eval(learner.save_checkpoint, learner.train_iter, collector_1011.envstep)
        if evaluator_89.should_eval(learner.train_iter):
            stop_89, _ = evaluator_89.eval(learner.save_checkpoint, learner.train_iter, collector_89.envstep)
        if evaluator_56.should_eval(learner.train_iter):
            stop_56, _ = evaluator_56.eval(learner.save_checkpoint, learner.train_iter, collector_56.envstep)
        if stop_1011 and stop_89 and stop_56:
            break
        # Collect data by default config n_sample/n_episode
        if seed == 0:
            new_data = collector_1011.collect(train_iter=learner.train_iter, policy_kwargs=collect_kwargs)
        elif seed == 1:
            new_data = collector_89.collect(train_iter=learner.train_iter, policy_kwargs=collect_kwargs)
        elif seed == 2:
            new_data = collector_56.collect(train_iter=learner.train_iter, policy_kwargs=collect_kwargs)
        else:
            raise
        seed_prob = level_sampler._sample_weights()
        tb_logger_1011.add_scalar("seeds/10m11m", seed_prob[0], global_step=learner.train_iter)
        tb_logger_1011.add_scalar("seeds/8m9m", seed_prob[1], global_step=learner.train_iter)
        tb_logger_1011.add_scalar("seeds/5m6m", seed_prob[2], global_step=learner.train_iter)
        # Learn policy from collected data
        learner.train(new_data, collector_1011.envstep)
        stacked_data = default_preprocess_learn(new_data, ignore_done=cfg_1011.policy.learn.ignore_done, use_nstep=False)
        stacked_data['seed'] = torch.ones((cfg_1011.policy.collect.n_sample), dtype=torch.float32) * seed
        level_sampler.update_with_rollouts(stacked_data, collector_env_num)
        seed = int(level_sampler.sample())

        if (collector_1011.envstep +  collector_89.envstep + collector_56.envstep) >= max_env_step or learner.train_iter >= max_train_iter:
            break

    # Learner's after_run hook.
    learner.call_hook('after_run')
    return policy