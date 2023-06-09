import torch
from .ppo_actor import PPOActor
from .ppo_critic import PPOCritic


class PPOPolicy:
    def __init__(self, args, obs_space, cent_obs_space, act_space, device=torch.device("cpu"), mode=0):
        self.args = args
        self.device = device
        # optimizer config
        self.lr = args.lr

        self.obs_space = obs_space
        self.cent_obs_space = cent_obs_space
        self.act_space = act_space

        self.actor = PPOActor(args, self.obs_space, self.act_space, self.device)
        self.critic = PPOCritic(args, self.cent_obs_space, self.device)
        self.mode = mode

        self.optimizer = torch.optim.Adam([
            {'params': self.actor.parameters()},
            {'params': self.critic.parameters()}
        ], lr=self.lr)

    def reset_lr(self, lr):
        self.lr = lr
        for p in self.optimizer.param_groups:
            p['lr'] = self.lr

    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks):
        """
        Returns:
            values, actions, action_log_probs, rnn_states_actor, rnn_states_critic
        """
        actions, action_log_probs, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, self.mode)
        values, rnn_states_critic = self.critic(cent_obs, rnn_states_critic, masks, self.mode)
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, cent_obs, rnn_states_critic, masks):
        """
        Returns:
            values
        """
        values, _ = self.critic(cent_obs, rnn_states_critic, masks, self.mode)
        return values

    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, action, masks, return_rnn=False):
        """
        Returns:
            values, action_log_probs, dist_entropy
        """

        if return_rnn == False:
            action_log_probs, dist_entropy = self.actor.evaluate_actions(obs, rnn_states_actor, action, masks,
                                                                         self.mode)
            values, _ = self.critic(cent_obs, rnn_states_critic, masks, self.mode)
            return values, action_log_probs, dist_entropy
        else:
            action_log_probs, dist_entropy, n_rnn_states_actor = self.actor.evaluate_actions(obs, rnn_states_actor,
                                                                                             action, masks, self.mode,
                                                                                             True)
            values, n_rnn_states_critic = self.critic(cent_obs, rnn_states_critic, masks, self.mode)
            return values, action_log_probs, dist_entropy, n_rnn_states_actor, n_rnn_states_critic

    def act(self, obs, rnn_states_actor, masks, deterministic=False):
        """
        Returns:
            actions, rnn_states_actor
        """
        actions, _, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, self.mode, deterministic)
        return actions, rnn_states_actor

    def prep_training(self):
        self.actor.train()
        self.critic.train()

    def prep_rollout(self):
        self.actor.eval()
        self.critic.eval()

    def copy(self):
        return PPOPolicy(self.args, self.obs_space, self.act_space, self.device)
