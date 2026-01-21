from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from project.train import train
from torch.profiler import ProfilerActivity, profile, record_function

# Pick profiler
profiler = "simple"  # can also pick 'simple', 'pytorch', 'advanced'

# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_stack=True) as prof:
with initialize(version_base=None, config_path="../configs"):
    cfg = compose(
        config_name="config",
        overrides=[
            "seed=123",
            "epochs=1",
            "wandb_project=null",
            "save_top_k=0",
            "+profiler_path=tests/profile/",
            f"+profiler_type={profiler}",
        ],
        return_hydra_config=True,
    )

    HydraConfig.instance().set_config(cfg)

train(cfg)
