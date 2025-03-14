from typing import Dict

import robomimic.models.base_nets as rmbn
import robomimic.scripts.generate_paper_configs as gpc
import robomimic.utils.obs_utils as ObsUtils
import torch
import torch.nn as nn
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
from robomimic.config import config_factory
from robomimic.scripts.generate_paper_configs import (
    modify_config_for_dataset,
    modify_config_for_default_image_exp,
    modify_config_for_default_low_dim_exp,
)

import diffusion_policy.model.vision.crop_randomizer as dmvc
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.diffusion.temporal_conditional_unet1d import (
    ConditionalUnet1D,
)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules
from diffusion_policy.rolling_diffusion import RollingDiffusion
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules


def get_robomimic_config(
    algo_name="bc_rnn", hdf5_type="low_dim", task_name="square", dataset_type="ph"
):
    base_dataset_dir = "/tmp/null"
    filter_key = None

    # decide whether to use low-dim or image training defaults
    modifier_for_obs = modify_config_for_default_image_exp
    if hdf5_type in ["low_dim", "low_dim_sparse", "low_dim_dense"]:
        modifier_for_obs = modify_config_for_default_low_dim_exp

    algo_config_name = "bc" if algo_name == "bc_rnn" else algo_name
    config = config_factory(algo_name=algo_config_name)
    # turn into default config for observation modalities (e.g.: low-dim or rgb)
    config = modifier_for_obs(config)
    # add in config based on the dataset
    config = modify_config_for_dataset(
        config=config,
        task_name=task_name,
        dataset_type=dataset_type,
        hdf5_type=hdf5_type,
        base_dataset_dir=base_dataset_dir,
        filter_key=filter_key,
    )
    # add in algo hypers based on dataset
    algo_config_modifier = getattr(gpc, f"modify_{algo_name}_config_for_dataset")
    config = algo_config_modifier(
        config=config,
        task_name=task_name,
        dataset_type=dataset_type,
        hdf5_type=hdf5_type,
    )
    return config


class UNetRollingDiffusionPolicy(BaseImagePolicy):
    def __init__(
        self,
        shape_meta: dict,
        horizon,
        num_clean,
        n_obs_steps,
        num_inference_steps=None,
        init_prob=0.1,
        obs_as_global_cond=True,
        crop_shape=(76, 76),
        diffusion_step_embed_dim=256,
        down_dims=(256, 512, 1024),
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=True,
        obs_encoder_group_norm=False,
        eval_fixed_crop=False,
        # parameters passed to step
        **kwargs,
    ):
        super().__init__()

        # parse shape_meta
        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta["obs"]
        obs_config = {"low_dim": [], "rgb": [], "depth": [], "scan": []}
        obs_key_shapes = dict()
        for key, attr in obs_shape_meta.items():
            shape = attr["shape"]
            obs_key_shapes[key] = list(shape)

            type = attr.get("type", "low_dim")
            if type == "rgb":
                obs_config["rgb"].append(key)
            elif type == "low_dim":
                obs_config["low_dim"].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        # get raw robomimic config
        config = get_robomimic_config(
            algo_name="bc_rnn", hdf5_type="image", task_name="square", dataset_type="ph"
        )

        with config.unlocked():
            # set config with shape_meta
            config.observation.modalities.obs = obs_config

            if crop_shape is None:
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == "CropRandomizer":
                        modality["obs_randomizer_class"] = None
            else:
                # set random crop parameter
                ch, cw = crop_shape
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == "CropRandomizer":
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw

        # init global state
        ObsUtils.initialize_obs_utils_with_config(config)

        # load model
        policy: PolicyAlgo = algo_factory(
            algo_name=config.algo_name,
            config=config,
            obs_key_shapes=obs_key_shapes,
            ac_dim=action_dim,
            device="cpu",
        )

        obs_encoder = policy.nets["policy"].nets["encoder"].nets["obs"]

        if obs_encoder_group_norm:
            # replace batch norm with group norm
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features // 16, num_channels=x.num_features
                ),
            )
            # obs_encoder.obs_nets['agentview_image'].nets[0].nets

        # obs_encoder.obs_randomizers['agentview_image']
        if eval_fixed_crop:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
                func=lambda x: dmvc.CropRandomizer(
                    input_shape=x.input_shape,
                    crop_height=x.crop_height,
                    crop_width=x.crop_width,
                    num_crops=x.num_crops,
                    pos_enc=x.pos_enc,
                ),
            )

        # create diffusion model
        obs_feature_dim = obs_encoder.output_shape()[0]
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps

        model = ConditionalUnet1D(
            input_dim=input_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = 100
        self.num_inference_steps = num_inference_steps

        self.diffusion = RollingDiffusion(
            model,
            num_features=action_dim,
            num_clean=num_clean,
            num_steps=num_inference_steps,
            window_size=horizon,
            init_prob=init_prob,
        )
        self.num_clean = num_clean

        print("Diffusion params: %e" % sum(p.numel() for p in self.model.parameters()))
        print(
            "Vision params: %e" % sum(p.numel() for p in self.obs_encoder.parameters())
        )

    def encode_obs(self, obs_dict):
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        To = self.n_obs_steps

        # handle different ways of passing observation
        global_cond = None

        # condition through global feature
        this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        # reshape back to B, Do
        global_cond = nobs_features.reshape(B, -1)
        # empty data for action

        return global_cond

    def reset(self):
        self.window = None

    def predict_action(
        self, obs_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert "past_action" not in obs_dict  # not implemented yet

        global_cond = self.encode_obs(obs_dict)

        if self.window is None:
            action_pred = self.diffusion.sample_init(global_cond)

            self.window = action_pred
        else:
            action_pred = self.diffusion.sample(self.window, global_cond)

            self.window = torch.cat(
                [action_pred[:, 1:], torch.randn_like(action_pred[:, :1])], dim=1
            )

        # unnormalize prediction
        action_pred = self.normalizer["action"].unnormalize(action_pred)

        # get action
        action = action_pred[:, :1]

        result = {"action": action, "action_pred": action_pred}
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        assert "valid_mask" not in batch
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])[
            :, self.n_obs_steps - 1 :
        ]
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(
                nobs, lambda x: x[:, : self.n_obs_steps, ...].reshape(-1, *x.shape[2:])
            )
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # Sample noise that we'll add to the images
        loss = self.diffusion(trajectory, global_cond)
        return loss
