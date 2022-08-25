from typing import Union, Optional, List, Any, Tuple
import os
import torch
import torch.nn as nn
from ditk import logging
from functools import partial
from tensorboardX import SummaryWriter

from ding.envs import get_vec_env_setting, create_env_manager
from ding.worker import BaseLearner, InteractionSerialMultiEnvEvaluator, BaseSerialCommander, create_buffer, \
    create_serial_collector
from ding.config import read_config, compile_config
from ding.policy import create_policy, PolicyFactory
from ding.reward_model import create_reward_model
from ding.utils import set_pkg_seed


def serial_pipeline_marine_onpolicy(
        input_cfg_5m6m: Union[str, Tuple[dict, dict]],
        input_cfg_8m9m: Union[str, Tuple[dict, dict]],
        input_cfg_10m11m: Union[str, Tuple[dict, dict]],
        seed: int = 0,
        env_setting: Optional[List[Any]] = None,
        model: Optional[torch.nn.Module] = None,
        max_train_iter: Optional[int] = int(1e10),
        max_env_step: Optional[int] = int(1e10),
) -> 'Policy':  # noqa
    """
    Overview:
        Serial pipeline entry on-policy RL for multi-map.
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
    if isinstance(input_cfg_5m6m, str):
        cfg_5m6m, create_cfg_5m6m = read_config(input_cfg_5m6m)
    else:
        cfg_5m6m, create_cfg_5m6m = input_cfg_5m6m
    if isinstance(input_cfg_8m9m, str):
        cfg_8m9m, create_cfg_8m9m = read_config(input_cfg_8m9m)
    else:
        cfg_8m9m, create_cfg_8m9m = input_cfg_8m9m
    if isinstance(input_cfg_10m11m, str):
        cfg_10m11m, create_cfg_10m11m = read_config(input_cfg_10m11m)
    else:
        cfg_10m11m, create_cfg_10m11m = input_cfg_10m11m
    create_cfg_5m6m.policy.type = create_cfg_5m6m.policy.type + '_command'
    create_cfg_8m9m.policy.type = create_cfg_8m9m.policy.type + '_command'
    create_cfg_10m11m.policy.type = create_cfg_10m11m.policy.type + '_command'
    env_fn = None if env_setting is None else env_setting[0]
    cfg_5m6m = compile_config(cfg_5m6m, seed=seed, env=env_fn, auto=True, create_cfg=create_cfg_5m6m, save_cfg=True)
    cfg_8m9m = compile_config(cfg_8m9m, seed=seed, env=env_fn, auto=True, create_cfg=create_cfg_8m9m, save_cfg=True)
    cfg_10m11m = compile_config(cfg_10m11m, seed=seed, env=env_fn, auto=True, create_cfg=create_cfg_10m11m,
                                save_cfg=True)
    # Create main components: env, policy
    if env_setting is None:
        env_fn_5m6m, collector_env_cfg_5m6m, evaluator_env_cfg_5m6m = get_vec_env_setting(cfg_5m6m.env)
        env_fn_8m9m, collector_env_cfg_8m9m, evaluator_env_cfg_8m9m = get_vec_env_setting(cfg_8m9m.env)
        env_fn_10m11m, collector_env_cfg_10m11m, evaluator_env_cfg_10m11m = get_vec_env_setting(cfg_10m11m.env)
    else:
        raise
        env_fn, collector_env_cfg, evaluator_env_cfg = env_setting
    # set default config for different workers with different envs
    collector_env_5m6m = create_env_manager(cfg_5m6m.env.manager,
                                            [partial(env_fn_5m6m, cfg=c) for c in collector_env_cfg_5m6m])
    collector_env_8m9m = create_env_manager(cfg_8m9m.env.manager,
                                            [partial(env_fn_8m9m, cfg=c) for c in collector_env_cfg_8m9m])
    collector_env_10m11m = create_env_manager(cfg_10m11m.env.manager,
                                              [partial(env_fn_10m11m, cfg=c) for c in collector_env_cfg_10m11m])
    evaluator_env_5m6m = create_env_manager(cfg_5m6m.env.manager,
                                            [partial(env_fn_5m6m, cfg=c) for c in evaluator_env_cfg_5m6m])
    evaluator_env_8m9m = create_env_manager(cfg_8m9m.env.manager,
                                            [partial(env_fn_8m9m, cfg=c) for c in evaluator_env_cfg_8m9m])
    evaluator_env_10m11m = create_env_manager(cfg_10m11m.env.manager,
                                              [partial(env_fn_10m11m, cfg=c) for c in evaluator_env_cfg_10m11m])
    collector_env_5m6m.seed(cfg_5m6m.seed)
    collector_env_8m9m.seed(cfg_8m9m.seed)
    collector_env_10m11m.seed(cfg_10m11m.seed)
    evaluator_env_5m6m.seed(cfg_5m6m.seed, dynamic_seed=False)
    evaluator_env_8m9m.seed(cfg_8m9m.seed, dynamic_seed=False)
    evaluator_env_10m11m.seed(cfg_10m11m.seed, dynamic_seed=False)
    set_pkg_seed(cfg_10m11m.seed, use_cuda=cfg_5m6m.policy.cuda)
    policy = create_policy(cfg_10m11m.policy, model=model, enable_field=['learn', 'collect', 'eval', 'command'])

    # Create worker components: learner, collector, evaluator, replay buffer, commander.
    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg_10m11m.exp_name), 'serial'))
    learner = BaseLearner(cfg_10m11m.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg_10m11m.exp_name)
    collector_5m6m = create_serial_collector(
        cfg_5m6m.policy.collect.collector,
        env=collector_env_5m6m,
        policy=policy.collect_mode,
        tb_logger=tb_logger,
        exp_name=cfg_5m6m.exp_name
    )
    collector_8m9m = create_serial_collector(
        cfg_8m9m.policy.collect.collector,
        env=collector_env_8m9m,
        policy=policy.collect_mode,
        tb_logger=tb_logger,
        exp_name=cfg_8m9m.exp_name
    )
    collector_10m11m = create_serial_collector(
        cfg_10m11m.policy.collect.collector,
        env=collector_env_10m11m,
        policy=policy.collect_mode,
        tb_logger=tb_logger,
        exp_name=cfg_10m11m.exp_name
    )
    evaluator_5m6m = InteractionSerialMultiEnvEvaluator(
        cfg_5m6m.policy.eval.evaluator, evaluator_env_5m6m, policy.eval_mode, tb_logger, exp_name=cfg_5m6m.exp_name
    )
    evaluator_8m9m = InteractionSerialMultiEnvEvaluator(
        cfg_8m9m.policy.eval.evaluator, evaluator_env_8m9m, policy.eval_mode, tb_logger, exp_name=cfg_8m9m.exp_name
    )
    evaluator_10m11m = InteractionSerialMultiEnvEvaluator(
        cfg_10m11m.policy.eval.evaluator, evaluator_env_10m11m, policy.eval_mode, tb_logger,
        exp_name=cfg_10m11m.exp_name
    )
    commander = BaseSerialCommander(
        cfg_10m11m.policy.other.commander, learner, collector_10m11m, evaluator_10m11m, None, policy.command_mode
    )

    # ==========
    # Main loop
    # ==========
    # Learner's before_run hook.
    learner.call_hook('before_run')

    # set target shape
    target_agent_num = 10
    target_action_shape = cfg_10m11m.policy.model.action_shape
    target_agent_obs_shape = cfg_10m11m.policy.model.agent_obs_shape
    target_global_obs_shape = cfg_10m11m.policy.model.global_obs_shape

    while True:
        collect_kwargs = commander.step()
        # Evaluate policy performance
        if evaluator_5m6m.should_eval(learner.train_iter):
            stop_5m6m, reward_5m6m = evaluator_5m6m.eval(learner.save_checkpoint, learner.train_iter,
                                                         collector_5m6m.envstep)
        if evaluator_8m9m.should_eval(learner.train_iter):
            stop_8m9m, reward_8m9m = evaluator_8m9m.eval(learner.save_checkpoint, learner.train_iter,
                                                         collector_8m9m.envstep)
        if evaluator_10m11m.should_eval(learner.train_iter):
            stop_10m11m, reward_10m11m = evaluator_10m11m.eval(learner.save_checkpoint, learner.train_iter,
                                                               collector_10m11m.envstep)
        if stop_5m6m and stop_8m9m and stop_10m11m:
            break
        # Collect data by default config n_sample/n_episode
        new_data_5m6m = collector_5m6m.collect(train_iter=learner.train_iter, policy_kwargs=collect_kwargs)
        new_data_8m9m = collector_8m9m.collect(train_iter=learner.train_iter, policy_kwargs=collect_kwargs)
        new_data_10m11m = collector_10m11m.collect(train_iter=learner.train_iter, policy_kwargs=collect_kwargs)

        def data_padding(ori_data):
            action_shape = ori_data[0]['logit'].shape[1]
            agent_num, agent_obs_shape = ori_data[0]['obs']['agent_state'].shape
            global_obs_shape = ori_data[0]['obs']['global_state'].shape[1]
            # each element in list
            for i in range(len(ori_data)):
                # pad the obs and next_obs
                for obs_key in ['obs', 'next_obs']:
                    ori_data[i][obs_key]['agent_state'] = nn.ConstantPad2d((0, target_agent_obs_shape-agent_obs_shape, 0, target_agent_num-agent_num), 0.)(ori_data[i][obs_key]['agent_state'])
                    ori_data[i][obs_key]['global_state'] = nn.ConstantPad2d((0, target_global_obs_shape-global_obs_shape, 0, target_agent_num-agent_num), 0.)(ori_data[i][obs_key]['global_state'])
                    ori_data[i][obs_key]['action_mask'] = nn.ConstantPad2d((0, target_action_shape-action_shape, 0, target_agent_num-agent_num), 0.)(ori_data[i][obs_key]['action_mask'])
                # pad the action
                ori_data[i]['action'] = nn.ConstantPad1d((0, target_agent_num-agent_num), 0.)(ori_data[i]['action'])
                # pad the logit
                ori_data[i]['logit'] = nn.ConstantPad2d((0, target_action_shape-action_shape, 0, target_agent_num-agent_num), 0.)(ori_data[i]['logit'])
                # pad the value
                ori_data[i]['value'] = nn.ConstantPad1d((0, target_agent_num-agent_num), 0.)(ori_data[i]['value'])
                # pad the adv
                ori_data[i]['adv'] = nn.ConstantPad1d((0, target_agent_num-agent_num), 0.)(ori_data[i]['adv'])
            return ori_data

        new_data_5m6m = data_padding(new_data_5m6m)
        new_data_8m9m = data_padding(new_data_8m9m)

        # Learn policy from collected data
        learner.train(new_data_5m6m, collector_5m6m.envstep)
        learner.train(new_data_8m9m, collector_8m9m.envstep)
        learner.train(new_data_10m11m, collector_10m11m.envstep)
        if collector_10m11m.envstep >= max_env_step or learner.train_iter >= max_train_iter:
            break

    # Learner's after_run hook.
    learner.call_hook('after_run')
    return policy
