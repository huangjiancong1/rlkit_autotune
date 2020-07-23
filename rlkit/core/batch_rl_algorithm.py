import abc

import gtimer as gt
from rlkit.core.rl_algorithm import BaseRLAlgorithm
from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.samplers.data_collector import PathCollector

import pandas as pd
from rlkit.core import logger
import math
import os
import numpy as np

class BatchRLAlgorithm(BaseRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
            self,
            automatic_policy_schedule,
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector: PathCollector,
            evaluation_data_collector: PathCollector,
            replay_buffer: ReplayBuffer,
            batch_size,
            max_path_length,
            num_epochs,
            num_eval_steps_per_epoch,
            num_expl_steps_per_train_loop,
            num_trains_per_train_loop,
            num_train_loops_per_epoch=1,
            min_num_steps_before_training=0,
    ):
        super().__init__(
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector,
            evaluation_data_collector,
            replay_buffer,
        )
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training
        self.automatic_policy_schedule = automatic_policy_schedule

    def _train(self):
        if self.min_num_steps_before_training > 0:
            init_expl_paths = self.expl_data_collector.collect_new_paths(
                self.max_path_length,
                self.min_num_steps_before_training,
                discard_incomplete_paths=False,
            )
            self.replay_buffer.add_paths(init_expl_paths)
            self.expl_data_collector.end_epoch(-1)

        for epoch in gt.timed_for(
                range(self._start_epoch, self.num_epochs),
                save_itrs=True,
        ): # episodes
            new_eval_paths = self.eval_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_eval_steps_per_epoch,
                discard_incomplete_paths=True,
            )        
            ## coverage evaluation recorder
            env_infos = []
            for eval_path in new_eval_paths:
                env_info = eval_path['env_infos']
                env_infos.append(env_info)
            try:
                if self.automatic_policy_schedule['test_coverage'] == True:
                    np.save(self.automatic_policy_schedule['vae_pkl_path']+'/evaluation/'+str(epoch)+'_evalation.npy', env_infos)
            except:
                pass

            gt.stamp('evaluation sampling')

            ## Automatic policy schedule
            if self.automatic_policy_schedule['use_automatic_schedule']:
                try:
                    progress_recorder = pd.read_csv(self.automatic_policy_schedule['vae_pkl_path']+'/progress.csv').to_dict('l')
                    if len(progress_recorder['vae_trainer/test/loss']) == 1:
                        minus_elbo = 0
                    else:

                        minus_elbo = progress_recorder['vae_trainer/test/loss'][-1]
                except:
                    minus_elbo = 0

                if epoch > 0:
                    if self.automatic_policy_schedule['automatic_policy_type'] == 'elbo': 
                        discount = self.automatic_policy_schedule['automatic_policy_discount']
                        self.num_trains_per_train_loop = int(discount*minus_elbo)

            for _ in range(self.num_train_loops_per_epoch): #1
                new_expl_paths = self.expl_data_collector.collect_new_paths(
                    self.max_path_length,
                    self.num_expl_steps_per_train_loop,
                    discard_incomplete_paths=False,
                )
                gt.stamp('exploration sampling', unique=False)

                self.replay_buffer.add_paths(new_expl_paths)
                gt.stamp('data storing', unique=False)

                self.training_mode(True)
                for _ in range(self.num_trains_per_train_loop):
                    train_data = self.replay_buffer.random_batch(
                        self.batch_size)
                    self.trainer.train(train_data)
                gt.stamp('training', unique=False)
                self.training_mode(False)

            logger.record_tabular('Policy Iteration', self.num_trains_per_train_loop)
            self._end_epoch(epoch)
