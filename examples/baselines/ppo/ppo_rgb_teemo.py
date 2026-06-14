# PPO RGB + TEEMO spatio-temporal semantic graph (StackCube-v1).
#
# Fork of examples/baselines/ppo/ppo_rgb.py. Drops the old prototype's
# stackcube_graph / graph_modules import and consumes the FULL TEEMO graph
# package at the repo root.
#
# Flags (default OFF -> behavior identical to stock ppo_rgb.py):
#   --use_graph_critic       graph latent concatenated to the critic input only
#   --use_graph_aux          per-relation CE heads on the obs latent (aux loss)
#   --use_causal_mask        Gumbel-Sigmoid binary mask over relations (+ L1)
#   --graph_encoder          'mlp' (over one-hot, fast) or 'gnn' (paper, message
#                            passing over the structured graph dict)
#   --graph_aux_coef         scale on the aux loss (default 0.1)
#   --mask_l1_coef           L1 sparsity on the causal mask probs (default 1e-3)
#   --graph_hidden           encoder hidden dim (default 128)
#   --graph_layers           gnn message-passing layers (default 2)
#   --no_z                   skip z (masked-RGB evidence); zeros in the schema
#   --affordance_dir         path to per-class affordance .npz (default
#                            ../../../teemo/affordances)
#   --thresholds_path        path to thresholds.json (default
#                            ../../../teemo/thresholds_default.json)
#
# The actor never sees the graph. Eval rollout uses get_action() and does NOT
# build a graph (the deployable policy uses normal obs only).
from collections import defaultdict
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

# Make `import teemo` resolve regardless of the cwd we were launched from.
# teemo/ lives at the repo root; this file is at examples/baselines/ppo/.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ManiSkill imports.
import mani_skill.envs  # noqa: F401
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import (
    FlattenActionSpaceWrapper,
    FlattenRGBDObservationWrapper,
)
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

# TEEMO imports. These are side-effect free unless a graph flag is enabled.
from teemo import vocab
from teemo.graph_builder import (
    GRAPH_DIM,
    PAIRS,
    SLOT_ORDER,
    TeemoGraphSpec,
    build_graph,
    extract_state,
    extract_z_features,
)
from teemo.history import GraphHistory
from teemo.affordance_use import AffordanceBank
from teemo.encoders import (
    CausalRelationMask,
    GraphAuxiliaryHeads,
    RelationGraphEncoder,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
@dataclass
class Args:
    exp_name: Optional[str] = None
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "ManiSkill"
    wandb_entity: Optional[str] = None
    wandb_group: str = "PPO-TEEMO"
    capture_video: bool = True
    save_model: bool = True
    evaluate: bool = False
    checkpoint: Optional[str] = None
    render_mode: str = "all"

    env_id: str = "StackCube-v1"
    include_state: bool = True
    total_timesteps: int = 10_000_000
    learning_rate: float = 3e-4
    num_envs: int = 256
    num_eval_envs: int = 8
    partial_reset: bool = True
    eval_partial_reset: bool = False
    num_steps: int = 50
    num_eval_steps: int = 50
    reconfiguration_freq: Optional[int] = None
    eval_reconfiguration_freq: Optional[int] = 1
    control_mode: Optional[str] = "pd_joint_delta_pos"
    anneal_lr: bool = False
    gamma: float = 0.8
    gae_lambda: float = 0.9
    num_minibatches: int = 32
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = False
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.2
    reward_scale: float = 1.0
    eval_freq: int = 25
    save_train_video_freq: Optional[int] = None
    finite_horizon_gae: bool = False

    # ---- TEEMO graph flags ------------------------------------------------
    use_graph_critic: bool = False
    """If True, concat a graph latent to the critic input. Actor is never given the graph."""
    use_graph_aux: bool = False
    """If True, train per-relation CE heads on the obs latent to predict graph labels."""
    use_causal_mask: bool = False
    """If True (only meaningful with --graph_encoder=gnn), insert a Gumbel-Sigmoid binary mask over relations + L1 sparsity penalty."""
    graph_encoder: str = "mlp"
    """'mlp' over the one-hot (fast) or 'gnn' (paper relation-aware message passing)."""
    graph_aux_coef: float = 0.1
    mask_l1_coef: float = 1e-3
    graph_hidden: int = 128
    graph_layers: int = 2
    graph_class_emb_dim: int = 8
    no_z: bool = False
    """Skip masked-RGB z extraction (z stays zero in the node schema)."""
    affordance_dir: Optional[str] = None
    """Path to teemo/affordances directory (per-class .npz). Defaults to <repo>/teemo/affordances."""
    thresholds_path: Optional[str] = None
    """Path to thresholds.json. Defaults to <repo>/teemo/thresholds_default.json (placeholder edges)."""

    # filled in at runtime
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


# ---------------------------------------------------------------------------
# Sidecar wrapper: stash raw sensor_data on the base env so segmentation
# survives FlattenRGBDObservationWrapper (which strips sensor_data entirely).
# ---------------------------------------------------------------------------
class SidecarSensorWrapper(gym.ObservationWrapper):
    """Side-channel for the raw camera sensor_data dict.

    Inserted between the base env and FlattenRGBDObservationWrapper. The
    observation is passed through unchanged; sensor_data is exposed via
    ``base_env._teemo_last_sensor_data`` so downstream code (e.g. teemo
    z-feature extraction) can still read segmentation after the flattener
    has dropped sensor_data from the obs dict.
    """

    def observation(self, observation):
        if isinstance(observation, dict) and "sensor_data" in observation:
            # Store a reference; FlattenRGBDObservationWrapper.observation() will
            # later pop sensor_data from the outer dict but the dict itself
            # remains alive via this reference.
            self.unwrapped._teemo_last_sensor_data = observation["sensor_data"]
        return observation


def _z_obs_view(base_env):
    sd = getattr(base_env, "_teemo_last_sensor_data", None)
    if sd is None:
        return None
    return {"sensor_data": sd}


# ---------------------------------------------------------------------------
# DictArray helper (verbatim from ppo_rgb.py)
# ---------------------------------------------------------------------------
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class DictArray(object):
    def __init__(self, buffer_shape, element_space, data_dict=None, device=None):
        self.buffer_shape = buffer_shape
        if data_dict:
            self.data = data_dict
        else:
            assert isinstance(element_space, gym.spaces.dict.Dict)
            self.data = {}
            for k, v in element_space.items():
                if isinstance(v, gym.spaces.dict.Dict):
                    self.data[k] = DictArray(buffer_shape, v, device=device)
                else:
                    dtype = (
                        torch.float32 if v.dtype in (np.float32, np.float64) else
                        torch.uint8 if v.dtype == np.uint8 else
                        torch.int16 if v.dtype == np.int16 else
                        torch.int32 if v.dtype == np.int32 else
                        v.dtype
                    )
                    self.data[k] = torch.zeros(buffer_shape + v.shape, dtype=dtype, device=device)

    def keys(self):
        return self.data.keys()

    def __getitem__(self, index):
        if isinstance(index, str):
            return self.data[index]
        return {k: v[index] for k, v in self.data.items()}

    def __setitem__(self, index, value):
        if isinstance(index, str):
            self.data[index] = value
        for k, v in value.items():
            self.data[k][index] = v

    @property
    def shape(self):
        return self.buffer_shape

    def reshape(self, shape):
        t = len(self.buffer_shape)
        new_dict = {}
        for k, v in self.data.items():
            if isinstance(v, DictArray):
                new_dict[k] = v.reshape(shape)
            else:
                new_dict[k] = v.reshape(shape + v.shape[t:])
        new_buffer_shape = next(iter(new_dict.values())).shape[:len(shape)]
        return DictArray(new_buffer_shape, None, data_dict=new_dict)


# ---------------------------------------------------------------------------
# Visual feature net (verbatim NatureCNN from ppo_rgb.py)
# ---------------------------------------------------------------------------
class NatureCNN(nn.Module):
    def __init__(self, sample_obs):
        super().__init__()
        extractors = {}
        self.out_features = 0
        feature_size = 256
        in_channels = sample_obs["rgb"].shape[-1]
        image_size = (sample_obs["rgb"].shape[1], sample_obs["rgb"].shape[2])

        cnn = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flatten = cnn(sample_obs["rgb"].float().permute(0, 3, 1, 2).cpu()).shape[1]
            fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
        extractors["rgb"] = nn.Sequential(cnn, fc)
        self.out_features += feature_size

        if "state" in sample_obs:
            state_size = sample_obs["state"].shape[-1]
            extractors["state"] = nn.Linear(state_size, 256)
            self.out_features += 256

        self.extractors = nn.ModuleDict(extractors)

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []
        for key, extractor in self.extractors.items():
            obs = observations[key]
            if key == "rgb":
                obs = obs.float().permute(0, 3, 1, 2) / 255.0
            encoded_tensor_list.append(extractor(obs))
        return torch.cat(encoded_tensor_list, dim=1)


# ---------------------------------------------------------------------------
# MLP graph encoder over the one-hot. Fast path; does not see node features.
# Used when --graph_encoder=mlp.
# ---------------------------------------------------------------------------
class OneHotGraphMLP(nn.Module):
    def __init__(self, graph_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(graph_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
        )
        self.out_dim = hidden

    def forward(self, graph_onehot: torch.Tensor) -> torch.Tensor:
        return self.net(graph_onehot)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
class Agent(nn.Module):
    """PPO actor-critic with optional TEEMO graph wiring.

    - Actor    : sees feature_net(obs) only. Policy is deployable from RGB+state.
    - Critic   : sees concat(obs_latent, graph_encoder(graph)) when --use_graph_critic.
    - Aux head : trained from obs_latent to predict per-relation labels when
                 --use_graph_aux. Gradient flows into feature_net.
    - Mask     : optional Gumbel-Sigmoid binary mask over relations when
                 --use_causal_mask (gnn encoder only). Adds L1 to the loss.
    """

    def __init__(
        self,
        envs,
        sample_obs,
        *,
        use_graph_critic: bool = False,
        use_graph_aux: bool = False,
        use_causal_mask: bool = False,
        graph_encoder: str = "mlp",
        graph_spec: Optional[TeemoGraphSpec] = None,
        node_in_dim: int = 0,
        graph_hidden: int = 128,
        graph_layers: int = 2,
    ):
        super().__init__()
        self.feature_net = NatureCNN(sample_obs=sample_obs)
        latent_size = self.feature_net.out_features

        self.use_graph_critic = use_graph_critic
        self.use_graph_aux = use_graph_aux
        self.use_causal_mask = use_causal_mask and use_graph_critic and graph_encoder == "gnn"
        self.graph_encoder_type = graph_encoder
        self.graph_spec = graph_spec

        if use_graph_critic:
            if graph_encoder == "gnn":
                assert graph_spec is not None and node_in_dim > 0
                self.graph_encoder = RelationGraphEncoder(
                    graph_spec, node_in_dim=node_in_dim,
                    hidden=graph_hidden, layers=graph_layers,
                )
                enc_out = self.graph_encoder.out_dim
            elif graph_encoder == "mlp":
                self.graph_encoder = OneHotGraphMLP(GRAPH_DIM, hidden=graph_hidden)
                enc_out = self.graph_encoder.out_dim
            else:
                raise ValueError(f"unknown --graph_encoder={graph_encoder}")
            critic_in = latent_size + enc_out
        else:
            self.graph_encoder = None
            critic_in = latent_size

        if self.use_causal_mask:
            assert graph_spec is not None
            self.causal_mask = CausalRelationMask(graph_spec, rel_feat_dim=GRAPH_DIM)
        else:
            self.causal_mask = None

        self.critic = nn.Sequential(
            layer_init(nn.Linear(critic_in, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, 1)),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(latent_size, 512)),
            nn.ReLU(inplace=True),
            layer_init(
                nn.Linear(512, int(np.prod(envs.unwrapped.single_action_space.shape))),
                std=0.01 * np.sqrt(2),
            ),
        )
        self.actor_logstd = nn.Parameter(
            torch.ones(1, int(np.prod(envs.unwrapped.single_action_space.shape))) * -0.5
        )

        if use_graph_aux:
            self.graph_aux = GraphAuxiliaryHeads(latent_size)
        else:
            self.graph_aux = None

    # --- encoder dispatch --------------------------------------------------
    def _encode_graph(self, graph_dict):
        """graph_dict: must contain 'onehot'; if encoder is gnn must also have
        'nodes' and 'targets'. Returns (graph_latent (N,enc_out), l1 (scalar))."""
        device = graph_dict["onehot"].device
        l1 = torch.zeros((), device=device)
        if self.use_causal_mask:
            mask, l1 = self.causal_mask(graph_dict["onehot"])
        else:
            mask = None
        if self.graph_encoder_type == "gnn":
            g = self.graph_encoder(graph_dict, mask=mask)
        else:
            g = self.graph_encoder(graph_dict["onehot"])
        return g, l1

    def _critic_from_latent(self, latent, graph_dict=None):
        if self.use_graph_critic:
            assert graph_dict is not None, "use_graph_critic=True but no graph dict was passed"
            g, _ = self._encode_graph(graph_dict)
            return self.critic(torch.cat([latent, g], dim=-1))
        return self.critic(latent)

    # --- public API --------------------------------------------------------
    def get_features(self, x):
        return self.feature_net(x)

    def get_value(self, x, graph_dict=None):
        latent = self.feature_net(x)
        return self._critic_from_latent(latent, graph_dict)

    def get_action(self, x, deterministic=False):
        x = self.feature_net(x)
        action_mean = self.actor_mean(x)
        if deterministic:
            return action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.sample()

    def get_action_and_value(self, x, action=None, graph_dict=None):
        latent = self.feature_net(x)
        action_mean = self.actor_mean(latent)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        value = self._critic_from_latent(latent, graph_dict)
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), value

    def get_action_value_and_aux(self, x, action=None, graph_dict=None):
        """Like get_action_and_value but also returns aux_logits and mask_l1.

        Aux logits come purely from the obs latent (no graph input). Mask L1 is
        zero unless --use_causal_mask + --graph_encoder=gnn + --use_graph_critic.
        """
        latent = self.feature_net(x)
        action_mean = self.actor_mean(latent)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        device = latent.device
        mask_l1 = torch.zeros((), device=device)
        if self.use_graph_critic:
            g, l1 = self._encode_graph(graph_dict)
            value = self.critic(torch.cat([latent, g], dim=-1))
            mask_l1 = l1
        else:
            value = self.critic(latent)
        aux_logits = self.graph_aux(latent) if (self.use_graph_aux and self.graph_aux is not None) else None
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), value, aux_logits, mask_l1


class Logger:
    def __init__(self, log_wandb=False, tensorboard: SummaryWriter = None) -> None:
        self.writer = tensorboard
        self.log_wandb = log_wandb

    def add_scalar(self, tag, scalar_value, step):
        if self.log_wandb:
            import wandb
            wandb.log({tag: scalar_value}, step=step)
        self.writer.add_scalar(tag, scalar_value, step)

    def close(self):
        self.writer.close()


# ---------------------------------------------------------------------------
# Helpers for graph dict reconstruction from rollout buffers
# ---------------------------------------------------------------------------
def make_zero_graph_dict(N, spec, node_feat_dim, device):
    return {
        "onehot": torch.zeros(N, GRAPH_DIM, device=device),
        "targets": {f"{p}:{f}": torch.zeros(N, dtype=torch.long, device=device)
                    for (p, f) in SLOT_ORDER},
        "nodes": {name: torch.zeros(N, node_feat_dim, device=device)
                  for name in spec.node_names},
    }


def gather_graph_dict(onehot_buf, targets_buf, nodes_buf, mb_inds, spec):
    """mb_inds: 1-D tensor of indices into the flattened batch dim."""
    return {
        "onehot": onehot_buf[mb_inds],
        "targets": {k: v[mb_inds] for k, v in targets_buf.items()},
        "nodes": {name: v[mb_inds] for name, v in nodes_buf.items()},
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name

    # Resolve teemo paths.
    if args.thresholds_path is None:
        args.thresholds_path = os.path.join(_REPO_ROOT, "teemo", "thresholds_default.json")
    if args.affordance_dir is None:
        args.affordance_dir = os.path.join(_REPO_ROOT, "teemo", "affordances")

    use_graph = args.use_graph_critic or args.use_graph_aux
    if use_graph:
        assert args.env_id == "StackCube-v1", (
            "TEEMO PAIRS are instantiated for StackCube-v1; got "
            f"env_id={args.env_id!r}. Add a task-specific pair set to extend."
        )
        if args.use_causal_mask and not args.use_graph_critic:
            print("[teemo] note: --use_causal_mask requires --use_graph_critic; the mask will be inactive.")
        if args.use_causal_mask and args.graph_encoder != "gnn":
            print("[teemo] note: --use_causal_mask only meaningful with --graph_encoder=gnn; the mask will be inactive.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # ---- env setup --------------------------------------------------------
    # Use rgb+segmentation when graph is on so teemo's z-feature extractor can
    # read masks; the SidecarSensorWrapper stashes sensor_data on base_env so
    # FlattenRGBDObservationWrapper can still drop it from the policy obs.
    obs_mode = "rgb+segmentation" if (use_graph and not args.no_z) else "rgb"
    env_kwargs = dict(obs_mode=obs_mode, render_mode=args.render_mode, sim_backend="physx_cuda")
    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode

    eval_envs = gym.make(args.env_id, num_envs=args.num_eval_envs,
                         reconfiguration_freq=args.eval_reconfiguration_freq, **env_kwargs)
    envs = gym.make(args.env_id, num_envs=args.num_envs if not args.evaluate else 1,
                    reconfiguration_freq=args.reconfiguration_freq, **env_kwargs)

    if use_graph and not args.no_z:
        envs = SidecarSensorWrapper(envs)
        eval_envs = SidecarSensorWrapper(eval_envs)

    envs = FlattenRGBDObservationWrapper(envs, rgb=True, depth=False, state=args.include_state)
    eval_envs = FlattenRGBDObservationWrapper(eval_envs, rgb=True, depth=False, state=args.include_state)

    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)
    if args.capture_video:
        eval_output_dir = f"runs/{run_name}/videos"
        if args.evaluate:
            eval_output_dir = f"{os.path.dirname(args.checkpoint)}/test_videos"
        print(f"Saving eval videos to {eval_output_dir}")
        if args.save_train_video_freq is not None:
            save_video_trigger = lambda x: (x // args.num_steps) % args.save_train_video_freq == 0
            envs = RecordEpisode(envs, output_dir=f"runs/{run_name}/train_videos",
                                 save_trajectory=False, save_video_trigger=save_video_trigger,
                                 max_steps_per_video=args.num_steps, video_fps=30)
        eval_envs = RecordEpisode(eval_envs, output_dir=eval_output_dir,
                                  save_trajectory=args.evaluate, trajectory_name="trajectory",
                                  max_steps_per_video=args.num_eval_steps, video_fps=30)
    envs = ManiSkillVectorEnv(envs, args.num_envs,
                              ignore_terminations=not args.partial_reset, record_metrics=True)
    eval_envs = ManiSkillVectorEnv(eval_envs, args.num_eval_envs,
                                   ignore_terminations=not args.eval_partial_reset, record_metrics=True)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_episode_steps = gym_utils.find_max_episode_steps_value(envs._env)
    logger = None
    if not args.evaluate:
        print("Running training")
        if args.track:
            import wandb
            config = vars(args)
            config["env_cfg"] = dict(**env_kwargs, num_envs=args.num_envs, env_id=args.env_id,
                                    reward_mode="normalized_dense", env_horizon=max_episode_steps,
                                    partial_reset=args.partial_reset)
            config["eval_env_cfg"] = dict(**env_kwargs, num_envs=args.num_eval_envs, env_id=args.env_id,
                                         reward_mode="normalized_dense", env_horizon=max_episode_steps,
                                         partial_reset=args.partial_reset)
            wandb.init(project=args.wandb_project_name, entity=args.wandb_entity,
                       sync_tensorboard=False, config=config, name=run_name,
                       save_code=True, group=args.wandb_group, tags=["ppo", "teemo"])
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        logger = Logger(log_wandb=args.track, tensorboard=writer)
    else:
        print("Running evaluation")

    # ---- rollout storage --------------------------------------------------
    obs = DictArray((args.num_steps, args.num_envs), envs.single_observation_space, device=device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # ---- TEEMO graph state ------------------------------------------------
    graph_spec = TeemoGraphSpec() if use_graph else None
    if use_graph:
        # Fixed (non-trainable) semantic-class embedding lookup.
        class_embed = nn.Embedding(len(vocab.SEMANTIC_CLASSES), args.graph_class_emb_dim).to(device)
        class_embed.weight.requires_grad_(False)

        affordance_bank = AffordanceBank(args.affordance_dir, device)
        with open(args.thresholds_path) as fth:
            thresholds = json.load(fth)

        graph_history = GraphHistory(args.num_envs, graph_spec.temporal_k, device)

        # Discover node feature dim by building one graph at the initial state.
        next_obs_init, _ = envs.reset(seed=args.seed)
        init_state = extract_state(envs.unwrapped)
        graph_history.push(init_state,
                           just_reset=torch.ones(args.num_envs, dtype=torch.bool, device=device))
        init_window, init_valid = graph_history.get_window()
        init_z = (extract_z_features(_z_obs_view(envs.unwrapped), envs.unwrapped, graph_spec)
                  if not args.no_z else {})
        with torch.no_grad():
            init_graph = build_graph(init_state, init_z, init_window, init_valid,
                                     thresholds, affordance_bank, graph_spec, class_embed)
        node_feat_dim = int(next(iter(init_graph["nodes"].values())).shape[-1])
        next_obs = next_obs_init

        graphs_onehot_buf = torch.zeros((args.num_steps, args.num_envs, GRAPH_DIM), device=device)
        graph_targets_buf = {
            f"{p}:{f}": torch.zeros((args.num_steps, args.num_envs), dtype=torch.long, device=device)
            for (p, f) in SLOT_ORDER
        }
        graph_nodes_buf = {
            name: torch.zeros((args.num_steps, args.num_envs, node_feat_dim), device=device)
            for name in graph_spec.node_names
        }
    else:
        class_embed = None
        affordance_bank = None
        thresholds = None
        graph_history = None
        node_feat_dim = 0
        graphs_onehot_buf = None
        graph_targets_buf = None
        graph_nodes_buf = None
        next_obs, _ = envs.reset(seed=args.seed)

    eval_obs, _ = eval_envs.reset(seed=args.seed)
    next_done = torch.zeros(args.num_envs, device=device)

    print(f"####")
    print(f"args.num_iterations={args.num_iterations} args.num_envs={args.num_envs} args.num_eval_envs={args.num_eval_envs}")
    print(f"args.minibatch_size={args.minibatch_size} args.batch_size={args.batch_size} args.update_epochs={args.update_epochs}")
    if use_graph:
        print(f"teemo: graph_dim={GRAPH_DIM} num_slots={len(SLOT_ORDER)} node_feat_dim={node_feat_dim} encoder={args.graph_encoder}")
    print(f"####")

    agent = Agent(
        envs, sample_obs=next_obs,
        use_graph_critic=args.use_graph_critic,
        use_graph_aux=args.use_graph_aux,
        use_causal_mask=args.use_causal_mask,
        graph_encoder=args.graph_encoder,
        graph_spec=graph_spec,
        node_in_dim=node_feat_dim,
        graph_hidden=args.graph_hidden,
        graph_layers=args.graph_layers,
    ).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    if args.checkpoint:
        agent.load_state_dict(torch.load(args.checkpoint))

    cumulative_times = defaultdict(float)
    global_step = 0
    start_time = time.time()

    for iteration in range(1, args.num_iterations + 1):
        print(f"Epoch: {iteration}, global_step={global_step}")
        final_values = torch.zeros((args.num_steps, args.num_envs), device=device)
        agent.eval()
        if iteration % args.eval_freq == 1:
            print("Evaluating")
            stime = time.perf_counter()
            eval_obs, _ = eval_envs.reset()
            eval_metrics = defaultdict(list)
            num_episodes = 0
            for _ in range(args.num_eval_steps):
                with torch.no_grad():
                    eval_obs, eval_rew, eval_terminations, eval_truncations, eval_infos = eval_envs.step(
                        agent.get_action(eval_obs, deterministic=True)
                    )
                    if "final_info" in eval_infos:
                        mask = eval_infos["_final_info"]
                        num_episodes += mask.sum()
                        for k, v in eval_infos["final_info"]["episode"].items():
                            eval_metrics[k].append(v)
            print(f"Evaluated {args.num_eval_steps * args.num_eval_envs} steps resulting in {num_episodes} episodes")
            for k, v in eval_metrics.items():
                mean = torch.stack(v).float().mean()
                if logger is not None:
                    logger.add_scalar(f"eval/{k}", mean, global_step)
                print(f"eval_{k}_mean={mean}")
            if logger is not None:
                eval_time = time.perf_counter() - stime
                cumulative_times["eval_time"] += eval_time
                logger.add_scalar("time/eval_time", eval_time, global_step)
            if args.evaluate:
                break
        if args.save_model and iteration % args.eval_freq == 1:
            model_path = f"runs/{run_name}/ckpt_{iteration}.pt"
            torch.save(agent.state_dict(), model_path)
            print(f"model saved to {model_path}")

        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        rollout_time = time.perf_counter()
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # Build the graph for the current next_obs. The history was last
            # pushed at the end of the previous step (or as the initial seed),
            # so the window's newest is the state matching next_obs.
            if use_graph:
                cur_state = extract_state(envs.unwrapped)
                cur_z = (extract_z_features(_z_obs_view(envs.unwrapped), envs.unwrapped, graph_spec)
                         if not args.no_z else {})
                window, valid = graph_history.get_window()
                with torch.no_grad():
                    graph = build_graph(cur_state, cur_z, window, valid,
                                        thresholds, affordance_bank, graph_spec, class_embed)
                graphs_onehot_buf[step] = graph["onehot"]
                for slot_key, t in graph["targets"].items():
                    graph_targets_buf[slot_key][step] = t
                for name, nf in graph["nodes"].items():
                    graph_nodes_buf[name][step] = nf
                cur_graph_dict = {"onehot": graph["onehot"],
                                  "targets": graph["targets"],
                                  "nodes": graph["nodes"]}
            else:
                cur_graph_dict = None

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(
                    next_obs,
                    graph_dict=cur_graph_dict if args.use_graph_critic else None,
                )
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(action)
            next_done = torch.logical_or(terminations, truncations).to(torch.float32)
            rewards[step] = reward.view(-1) * args.reward_scale

            # Push the post-step state into the graph history (consistent with
            # the new next_obs). just_reset clears history for any env that
            # auto-reset so temporal labels never cross episode boundaries.
            if use_graph:
                new_state = extract_state(envs.unwrapped)
                graph_history.push(new_state,
                                   just_reset=torch.logical_or(terminations, truncations))

            if "final_info" in infos:
                final_info = infos["final_info"]
                done_mask = infos["_final_info"]
                for k, v in final_info["episode"].items():
                    logger.add_scalar(f"train/{k}", v[done_mask].float().mean(), global_step)

                for k in infos["final_observation"]:
                    infos["final_observation"][k] = infos["final_observation"][k][done_mask]
                with torch.no_grad():
                    if args.use_graph_critic:
                        # Done envs lost the privileged final state to auto-reset,
                        # so we bootstrap with a zero graph. The mismatch only
                        # affects the GAE target for the terminal step and is
                        # bounded by next_not_done in the GAE pass.
                        n_done = int(done_mask.sum().item())
                        zero_g = make_zero_graph_dict(n_done, graph_spec, node_feat_dim, device)
                        final_values[step,
                                     torch.arange(args.num_envs, device=device)[done_mask]] = (
                            agent.get_value(infos["final_observation"], graph_dict=zero_g).view(-1)
                        )
                    else:
                        final_values[step,
                                     torch.arange(args.num_envs, device=device)[done_mask]] = (
                            agent.get_value(infos["final_observation"]).view(-1)
                        )

        rollout_time = time.perf_counter() - rollout_time
        cumulative_times["rollout_time"] += rollout_time

        # ---- GAE bootstrap ------------------------------------------------
        with torch.no_grad():
            if args.use_graph_critic:
                post_state = extract_state(envs.unwrapped)
                post_z = (extract_z_features(_z_obs_view(envs.unwrapped), envs.unwrapped, graph_spec)
                          if not args.no_z else {})
                post_window, post_valid = graph_history.get_window()
                post_graph = build_graph(post_state, post_z, post_window, post_valid,
                                         thresholds, affordance_bank, graph_spec, class_embed)
                next_value = agent.get_value(next_obs, graph_dict={
                    "onehot": post_graph["onehot"],
                    "targets": post_graph["targets"],
                    "nodes": post_graph["nodes"],
                }).reshape(1, -1)
            else:
                next_value = agent.get_value(next_obs).reshape(1, -1)

            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    next_not_done = 1.0 - next_done
                    nextvalues = next_value
                else:
                    next_not_done = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                real_next_values = next_not_done * nextvalues + final_values[t]
                if args.finite_horizon_gae:
                    if t == args.num_steps - 1:
                        lam_coef_sum = 0.
                        reward_term_sum = 0.
                        value_term_sum = 0.
                    lam_coef_sum = lam_coef_sum * next_not_done
                    reward_term_sum = reward_term_sum * next_not_done
                    value_term_sum = value_term_sum * next_not_done
                    lam_coef_sum = 1 + args.gae_lambda * lam_coef_sum
                    reward_term_sum = args.gae_lambda * args.gamma * reward_term_sum + lam_coef_sum * rewards[t]
                    value_term_sum = args.gae_lambda * args.gamma * value_term_sum + args.gamma * real_next_values
                    advantages[t] = (reward_term_sum + value_term_sum) / lam_coef_sum - values[t]
                else:
                    delta = rewards[t] + args.gamma * real_next_values - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * next_not_done * lastgaelam
            returns = advantages + values

        # ---- flatten batch ------------------------------------------------
        b_obs = obs.reshape((-1,))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        if use_graph:
            b_graph_oh = graphs_onehot_buf.reshape(-1, GRAPH_DIM)
            b_graph_targets = {k: v.reshape(-1) for k, v in graph_targets_buf.items()}
            b_graph_nodes = {name: v.reshape(-1, node_feat_dim) for name, v in graph_nodes_buf.items()}

            # Cheap per-iteration sanity asserts (matches the old prototype's
            # contract). Shapes, one-hot group sums, target ranges.
            assert b_graph_oh.shape == (args.batch_size, GRAPH_DIM)
            off = 0
            for (pkey, fld) in SLOT_ORDER:
                nc = vocab.RELATION_NUM_CLASSES[fld]
                group_sum = b_graph_oh[:, off:off + nc].sum(dim=-1)
                assert torch.allclose(group_sum, torch.ones_like(group_sum)), (
                    f"one-hot group {pkey}:{fld} does not sum to 1"
                )
                off += nc
            assert off == GRAPH_DIM
            for slot_key, t_flat in b_graph_targets.items():
                fld = slot_key.split(":")[1]
                nc = vocab.RELATION_NUM_CLASSES[fld]
                if t_flat.numel() > 0:
                    assert int(t_flat.min().item()) >= 0 and int(t_flat.max().item()) < nc, (
                        f"target {slot_key} out of range [0,{nc}): "
                        f"min={int(t_flat.min().item())} max={int(t_flat.max().item())}"
                    )
        else:
            b_graph_oh = None
            b_graph_targets = None
            b_graph_nodes = None

        # ---- update -------------------------------------------------------
        agent.train()
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        aux_loss_running = 0.0
        aux_loss_count = 0
        mask_l1_running = 0.0
        mask_l1_count = 0
        update_time = time.perf_counter()
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                mb_inds_t = torch.as_tensor(mb_inds, device=device, dtype=torch.long)

                if args.use_graph_critic:
                    mb_graph_dict = gather_graph_dict(
                        b_graph_oh, b_graph_targets, b_graph_nodes, mb_inds_t, graph_spec
                    )
                else:
                    mb_graph_dict = None

                _, newlogprob, entropy, newvalue, aux_logits, mask_l1 = agent.get_action_value_and_aux(
                    b_obs[mb_inds],
                    b_actions[mb_inds],
                    graph_dict=mb_graph_dict,
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                if args.use_graph_aux and aux_logits is not None:
                    mb_targets = {k: v[mb_inds] for k, v in b_graph_targets.items()}
                    aux_loss = GraphAuxiliaryHeads.loss(aux_logits, mb_targets)
                    assert torch.isfinite(aux_loss), "graph_aux_loss is not finite"
                    loss = loss + args.graph_aux_coef * aux_loss
                    aux_loss_running += aux_loss.item()
                    aux_loss_count += 1

                if agent.use_causal_mask:
                    loss = loss + args.mask_l1_coef * mask_l1
                    mask_l1_running += mask_l1.item()
                    mask_l1_count += 1

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break
        update_time = time.perf_counter() - update_time
        cumulative_times["update_time"] += update_time

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        logger.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        logger.add_scalar("charts/use_graph_critic", float(args.use_graph_critic), global_step)
        logger.add_scalar("charts/use_graph_aux", float(args.use_graph_aux), global_step)
        logger.add_scalar("charts/use_causal_mask", float(args.use_causal_mask), global_step)
        logger.add_scalar("losses/value_loss", v_loss.item(), global_step)
        logger.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        logger.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        logger.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        logger.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        logger.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        logger.add_scalar("losses/explained_variance", explained_var, global_step)
        if args.use_graph_aux and aux_loss_count > 0:
            logger.add_scalar("losses/graph_aux_loss", aux_loss_running / aux_loss_count, global_step)
        if agent.use_causal_mask and mask_l1_count > 0:
            logger.add_scalar("losses/mask_l1", mask_l1_running / mask_l1_count, global_step)

        if use_graph:
            # Telemetry: per-class frequency for the 7-way continuous temporal
            # heads (where 'stable' can dominate and mask learning failure).
            for slot_key in ("tcp-cubeA:distance_change",
                             "tcp-cubeA:height_change",
                             "tcp-cubeA:alignment_change",
                             "grip-cubeA:aperture_change"):
                t_flat = graph_targets_buf[slot_key].reshape(-1)
                fld = slot_key.split(":")[1]
                for c in range(vocab.RELATION_NUM_CLASSES[fld]):
                    freq = (t_flat == c).float().mean().item()
                    logger.add_scalar(f"graph_class_freq/{slot_key}/class_{c}", freq, global_step)

        sps = int(global_step / max(time.time() - start_time, 1e-3))
        print("SPS:", sps)
        logger.add_scalar("charts/SPS", sps, global_step)
        logger.add_scalar("time/step", global_step, global_step)
        logger.add_scalar("time/update_time", update_time, global_step)
        logger.add_scalar("time/rollout_time", rollout_time, global_step)
        logger.add_scalar("time/rollout_fps", args.num_envs * args.num_steps / max(rollout_time, 1e-9), global_step)
        for k, v in cumulative_times.items():
            logger.add_scalar(f"time/total_{k}", v, global_step)
        logger.add_scalar("time/total_rollout+update_time",
                          cumulative_times["rollout_time"] + cumulative_times["update_time"], global_step)

    if args.save_model and not args.evaluate:
        model_path = f"runs/{run_name}/final_ckpt.pt"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

    envs.close()
    if logger is not None:
        logger.close()
