
import os
import argparse
import yaml


import torch
from torch.utils.tensorboard import SummaryWriter

from ml_logger import RUN, logger

import diffuser.utils as utils

from diffuser.utils.launcher_util import (
    build_config_from_dict,
    set_config_override_args,
    update_exp_spec_dict, 
    discover_latest_checkpoint_path,
)


def main(Config, RUN, tb_logger):
    
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    utils.set_seed(Config.seed)
    dataset_extra_kwargs = dict()

    # configs that does not exist in old yaml files
    Config.discrete_action = getattr(Config, "discrete_action", False)
    Config.state_loss_weight = getattr(Config, "state_loss_weight", None)
    Config.opponent_loss_weight = getattr(Config, "opponent_loss_weight", None)
    Config.use_seed_dataset = getattr(Config, "use_seed_dataset", False)
    
    Config.residual_attn = getattr(Config, "residual_attn", True)
    Config.use_temporal_attention = getattr(Config, "use_temporal_attention", True)
    
    Config.use_hetgat_attention = getattr(Config, "use_hetgat_attention", False)
    
    Config.env_ts_condition = getattr(Config, "env_ts_condition", False)
    Config.use_return_to_go = getattr(Config, "use_return_to_go", False)
    
    Config.joint_inv = getattr(Config, "joint_inv", False)
    Config.share_inv = getattr(Config, "share_inv", False)
    Config.het_inv = getattr(Config, "het_inv", False) 
    
    Config.graph_condition = getattr(Config, "graph_condition", False)
        
    Config.use_zero_padding = getattr(Config, "use_zero_padding", True)
    Config.use_inv_dyn = getattr(Config, "use_inv_dyn", True)
    Config.pred_future_padding = getattr(Config, "pred_future_padding", False)

    Config.blind_own_type = getattr(Config, "blind_own_type", False)

    if Config.blind_own_type:
        print("Agent's own type in observation is blinded")
    else:
        print("Agent's own type in observation is visible")
    
    if not hasattr(Config, "agent_condition_type"):
        if Config.decentralized_execution:
            Config.agent_condition_type = "single"
        else:
            Config.agent_condition_type = "all"


    # -----------------------------------------------------------------------------#
    # ---------------------------------- dataset ----------------------------------#
    # -----------------------------------------------------------------------------#
    dataset_config = utils.Config(
        Config.loader,
        savepath="dataset_config.pkl",
        env_type=Config.env_type,
        env=Config.dataset,
        n_agents=Config.n_agents,
        horizon=Config.horizon,
        history_horizon=Config.history_horizon,
        normalizer=Config.normalizer,
        preprocess_fns=Config.preprocess_fns,
        max_n_episodes=Config.max_n_episodes,
        use_padding=Config.use_padding,
        use_action=Config.use_action,
        discrete_action=Config.discrete_action,
        max_path_length=Config.max_path_length,
        include_returns=Config.returns_condition,
        include_env_ts=Config.env_ts_condition,
        returns_scale=Config.returns_scale,
        discount=Config.discount,
        termination_penalty=Config.termination_penalty,
        agent_share_parameters=utils.config.import_class(Config.model).agent_share_parameters,
        use_seed_dataset=Config.use_seed_dataset,
        seed=Config.seed,
        use_inv_dyn=Config.use_inv_dyn,
        decentralized_execution=Config.decentralized_execution,
        use_zero_padding=Config.use_zero_padding,
        agent_condition_type=Config.agent_condition_type,
        pred_future_padding=Config.pred_future_padding,
        debugging=getattr(Config, "debugging", False), 
        use_hetgat_attention=Config.use_hetgat_attention,
        blind_own_type=Config.blind_own_type,
        **dataset_extra_kwargs,
    )

    render_config = utils.Config(
        Config.renderer,
        savepath="render_config.pkl",
        env_type=Config.env_type,
        env=Config.dataset,
    )
    
    data_encoder_config = utils.Config(
        getattr(Config, "data_encoder", "utils.IdentityEncoder"),
        savepath="data_encoder_config.pkl",
    )

    dataset = dataset_config()
    renderer = render_config()
    data_encoder = data_encoder_config()
    
    observation_dim = dataset.observation_dim
    action_dim = dataset.action_dim

    # -----------------------------------------------------------------------------#
    # ------------------------------ model & trainer ------------------------------#
    # -----------------------------------------------------------------------------#
    model_config = utils.Config(
        Config.model,
        savepath="model_config.pkl",
        n_agents=Config.n_agents,
        horizon=Config.horizon + Config.history_horizon,
        history_horizon=Config.history_horizon,
        transition_dim=observation_dim,
        dim_mults=Config.dim_mults,
        returns_condition=Config.returns_condition,
        
        graph_condition=getattr(Config, "graph_condition", False),
        
        env_ts_condition=Config.env_ts_condition,
        dim=Config.dim,
        condition_dropout=Config.condition_dropout,
        residual_attn=Config.residual_attn,
        max_path_length=Config.max_path_length,
        use_temporal_attention=Config.use_temporal_attention,
        device=Config.device,
    )

    diffusion_config = utils.Config(
        Config.diffusion,
        savepath="diffusion_config.pkl",
        n_agents=Config.n_agents,
        horizon=Config.horizon,
        history_horizon=Config.history_horizon,
        observation_dim=observation_dim,
        action_dim=action_dim,
        discrete_action=Config.discrete_action,
        num_actions=getattr(dataset.env, "num_actions", 0),
        n_timesteps=Config.n_diffusion_steps,
        clip_denoised=Config.clip_denoised,
        predict_epsilon=Config.predict_epsilon,
        hidden_dim=Config.hidden_dim,
        train_only_inv=Config.train_only_inv,
        
        share_inv=Config.share_inv,
        joint_inv=Config.joint_inv,
        het_inv=Config.het_inv,
        
        # loss weighting
        action_weight=Config.action_weight,
        loss_weights=Config.loss_weights,
        state_loss_weight=Config.state_loss_weight,
        opponent_loss_weight=Config.opponent_loss_weight,
        loss_discount=Config.loss_discount,
        returns_condition=Config.returns_condition,
        graph_condition=getattr(Config, "graph_condition", False),
        condition_guidance_w=Config.condition_guidance_w,
        data_encoder=data_encoder,
        use_inv_dyn=Config.use_inv_dyn,
        device=Config.device,
    )

    trainer_config = utils.Config(
        utils.Trainer,
        savepath="trainer_config.pkl",
        train_batch_size=Config.batch_size,
        train_lr=Config.learning_rate,
        gradient_accumulate_every=Config.gradient_accumulate_every,
        ema_decay=Config.ema_decay,
        sample_freq=Config.sample_freq,
        save_freq=Config.save_freq,
        log_freq=Config.log_freq,
        label_freq=int(Config.n_train_steps // Config.n_saves),
        eval_freq=Config.eval_freq,
        save_parallel=Config.save_parallel,
        bucket=logger.root,
        n_reference=Config.n_reference,
        train_device=Config.device,
        save_checkpoints=Config.save_checkpoints,
        use_hetgat_attention=Config.use_hetgat_attention, 
    )

    evaluator_config = utils.Config(Config.evaluator, savepath="evaluator_config.pkl")

    
    
    # -----------------------------------------------------------------------------#
    # -------------------------------- instantiate --------------------------------#
    # -----------------------------------------------------------------------------#

    model = model_config()
    diffusion = diffusion_config(model)
    trainer = trainer_config(diffusion, dataset, renderer, tb_logger=tb_logger)

    if Config.eval_freq > 0:
        evaluator = evaluator_config(tb_logger=tb_logger)
        
        evaluator.init(eval_experiment_name=Config.exp_name, 
                       log_dir=f'{logger.root}/{logger.prefix}',
                       eval_env=Config.eval_env, 
                       num_eval=Config.num_eval,
                       num_envs=getattr(Config, "num_envs", Config.num_eval), 
                       condition_guidance_w=getattr(Config, "condition_guidance_w", None), 
                       use_ddim_sample=True,
                       n_ddim_steps=Config.n_ddim_steps,
                       )
        
        trainer.set_evaluator(evaluator)

    if Config.continue_training:
        
        loadpath = discover_latest_checkpoint_path(os.path.join(trainer.bucket, logger.prefix, "checkpoint"))
        
        if loadpath is not None:
            state_dict = torch.load(loadpath, map_location=Config.device)
            logger.print(f"\nLoaded checkpoint from {loadpath} (step {state_dict['step']})\n", color="green",)
            trainer.step = state_dict["step"]
            trainer.model.load_state_dict(state_dict["model"])
            trainer.ema_model.load_state_dict(state_dict["ema"])

    # -----------------------------------------------------------------------------#
    # ------------------------ test forward & backward pass -----------------------#
    # -----------------------------------------------------------------------------#

    utils.report_parameters(model)

    logger.print("Testing forward...", end=" ", flush=True)
    batch = utils.batchify(dataset[0], Config.device)
    loss, _ = diffusion.loss(**batch)
    loss.backward()
    logger.print("✓")

    # -----------------------------------------------------------------------------#
    # --------------------------------- main loop ---------------------------------#
    # -----------------------------------------------------------------------------#

    n_epochs = int((Config.n_train_steps - trainer.step) // Config.n_steps_per_epoch)

    for i in range(n_epochs):
        logger.print(f"Epoch {i} / {n_epochs} | {logger.prefix}")
        
        trainer.train(n_train_steps=Config.n_steps_per_epoch)
        
    trainer.finish_training()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment", help="experiment specification file")
    args, override_args = parser.parse_known_args()

    with open(args.experiment, "r") as spec_file:
        spec_string = spec_file.read()
        exp_specs = yaml.load(spec_string, Loader=yaml.SafeLoader)

    Config = build_config_from_dict(exp_specs)
    Config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Config, override_args_dict = set_config_override_args(Config, override_args)
    exp_specs = update_exp_spec_dict(exp_specs, override_args_dict)
    
    job_name = Config.job_name.format(**vars(Config))
    RUN.prefix, RUN.job_name, _ = RUN(
        script_path=__file__,
        exp_name=exp_specs["exp_name"],
        job_name=job_name + f"/{Config.seed}",
    )

    logger.configure(RUN.prefix, root=RUN.script_root)
    # logger.remove('*.pkl')
    logger.remove("traceback.err")
    logger.remove("parameters.pkl")
    logger.log_params(Config=vars(Config), RUN=vars(RUN))
    logger.log_text(
        """
                    charts:
                    - yKey: loss
                      xKey: steps
                    - yKey: a0_loss
                      xKey: steps
                    """,
        filename=".charts.yml",
        dedent=True,
        overwrite=True,
    )
    
    logger.save_yaml(exp_specs, "exp_specs.yml")
    tb_logger = SummaryWriter(log_dir=os.path.join(RUN.script_root, RUN.prefix))
    
    main(Config, RUN, tb_logger)
