import torch
import torch.nn as nn

from RLAlg.normalizer import Normalizer
from RLAlg.nn.layers import GRULayer, MLPLayer, GaussianHead, NormPosition
from RLAlg.nn.steps import StochasticContinuousPolicyStep

from .adain import AdaINBlock, AdaINResBlock


def _normalize_observation(
    normalizer: Normalizer,
    obs: torch.Tensor,
    update_normalizer: bool = False,
) -> torch.Tensor:
    if obs.ndim == 2:
        return normalizer(obs, update=update_normalizer)
    if obs.ndim == 3:
        flat_obs = obs.reshape(-1, obs.shape[-1])
        flat_obs = normalizer(flat_obs, update=update_normalizer)
        return flat_obs.reshape(*obs.shape)
    raise ValueError(f"Expected observation rank 2 or 3, got shape {tuple(obs.shape)}.")

class VanilaActor(nn.Module):
    def __init__(self, obs_dim:int, action_dim:int):
        super().__init__()

        self.normlizer = Normalizer((obs_dim,))

        self.encoder = nn.Sequential(
            MLPLayer(obs_dim, 2048, nn.SiLU(), NormPosition.POST),
            MLPLayer(2048, 1024, nn.SiLU(), NormPosition.POST),
            MLPLayer(1024, 512, nn.SiLU(), NormPosition.POST),
        )

        self.head = GaussianHead(512, action_dim)

    def forward(self, obs_dict:dict[str, torch.Tensor], action:torch.Tensor|None=None, update_normlizer:bool=False) -> StochasticContinuousPolicyStep:
        obs = obs_dict["obs"]
        obs = self.normlizer(obs, update=update_normlizer)
        x = self.encoder(obs)
        step = self.head(x, action)

        return step


class RecurrentActor(nn.Module):
    DEFAULT_HIDDEN_SIZE = 512
    DEFAULT_NUM_LAYERS = 1

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_size: int = DEFAULT_HIDDEN_SIZE,
        num_layers: int = DEFAULT_NUM_LAYERS,
    ):
        super().__init__()

        if hidden_size < 1:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}.")
        if num_layers < 1:
            raise ValueError(f"num_layers must be positive, got {num_layers}.")

        self.normlizer = Normalizer((obs_dim,))
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.encoder = nn.Sequential(
            MLPLayer(obs_dim, 2048, nn.SiLU(), NormPosition.POST),
            MLPLayer(2048, 1024, nn.SiLU(), NormPosition.POST),
            MLPLayer(1024, 512, nn.SiLU(), NormPosition.POST),
        )
        self.gru = GRULayer(512, hidden_size, num_layers=num_layers)
        self.head = GaussianHead(hidden_size, action_dim)

    def get_initial_state(
        self,
        batch_size: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        ref_param = self.head.mu_layer.weight
        return torch.zeros(
            self.num_layers,
            batch_size,
            self.hidden_size,
            device=device or ref_param.device,
            dtype=dtype or ref_param.dtype,
        )

    def forward(
        self,
        obs_dict: dict[str, torch.Tensor],
        action: torch.Tensor | None = None,
        initial_state: torch.Tensor | None = None,
        episode_starts: torch.Tensor | None = None,
        update_normlizer: bool = False,
    ) -> tuple[StochasticContinuousPolicyStep, torch.Tensor]:
        obs = obs_dict["obs"]
        obs = _normalize_observation(self.normlizer, obs, update_normlizer)
        x = self.encoder(obs)
        x, next_state = self.gru(x, hidden_state=initial_state, episode_starts=episode_starts)
        step = self.head(x, action)

        return step, next_state
    
class SplitEncoderActor(nn.Module):
    def __init__(self, robot_obs_dim:int, motion_obs_dim:int, action_dim:int):
        super().__init__()

        self.robot_obs_normlizer = Normalizer((robot_obs_dim,))
        self.motion_obs_normlizer = Normalizer((motion_obs_dim,))

        self.robot_encoder = MLPLayer(robot_obs_dim, 256, nn.Identity())
        self.motion_encoder = MLPLayer(motion_obs_dim, 256, nn.Identity())
        

        self.encoder = nn.Sequential(
            MLPLayer(512, 512, nn.SiLU(), NormPosition.POST),
            MLPLayer(512, 512, nn.SiLU(), NormPosition.POST),
            MLPLayer(512, 512, nn.SiLU(), NormPosition.POST),
        )

        self.head = GaussianHead(512, action_dim)

    def forward(self, obs_dict:dict[str, torch.Tensor], action:torch.Tensor|None=None, update_normlizer:bool=False) -> StochasticContinuousPolicyStep:
        robot_obs = obs_dict["robot_obs"]
        motion_obs = obs_dict["motion_obs"]

        robot_obs = self.robot_obs_normlizer(robot_obs, update_normlizer)
        motion_obs = self.motion_obs_normlizer(motion_obs, update_normlizer)

        x_robot = self.robot_encoder(robot_obs)
        x_motion = self.motion_encoder(motion_obs)

        x = torch.cat([x_robot, x_motion], dim=-1)

        x = self.encoder(x)
        step = self.head(x, action)

        return step

class AdaINActor(nn.Module):
    def __init__(self, robot_obs_dim:int, motion_obs_dim:int, action_dim:int):
        super().__init__()

        self.robot_obs_normlizer = Normalizer((robot_obs_dim,))
        self.motion_obs_normlizer = Normalizer((motion_obs_dim,))

        self.robot_encoder = MLPLayer(robot_obs_dim, 512, nn.Identity())
        self.motion_encoder = MLPLayer(motion_obs_dim, 512, nn.Identity())
        
        self.block_1 = AdaINBlock(512, 512)
        self.block_2 = AdaINBlock(512, 512)
        self.block_3 = AdaINBlock(512, 512)

        self.head = GaussianHead(512, action_dim)

    def forward(self, obs_dict:dict[str, torch.Tensor], action:torch.Tensor|None=None, update_normlizer:bool=False) -> StochasticContinuousPolicyStep:
        robot_obs = obs_dict["robot_obs"]
        motion_obs = obs_dict["motion_obs"]

        robot_obs = self.robot_obs_normlizer(robot_obs, update_normlizer)
        motion_obs = self.motion_obs_normlizer(motion_obs, update_normlizer)

        x_robot = self.robot_encoder(robot_obs)
        x_motion = self.motion_encoder(motion_obs)

        x = self.block_1(x_robot, x_motion)
        x = self.block_2(x, x_motion)
        x = self.block_3(x, x_motion)
        
        step = self.head(x, action)

        return step

class AdaINResActor(nn.Module):
    def __init__(
        self,
        robot_obs_dim: int,
        motion_obs_dim: int,
        action_dim: int,
        num_blocks: int = 3,
    ):
        super().__init__()

        if num_blocks < 1:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}.")

        self.robot_obs_normlizer = Normalizer((robot_obs_dim,))
        self.motion_obs_normlizer = Normalizer((motion_obs_dim,))
        self.num_blocks = num_blocks

        self.robot_encoder = nn.Sequential(
            MLPLayer(robot_obs_dim, 512, nn.SiLU(), NormPosition.POST),
            MLPLayer(512, 512, nn.SiLU(), NormPosition.POST),
            MLPLayer(512, 512, nn.Identity()),
        )

        self.motion_encoder = nn.Sequential(
            MLPLayer(motion_obs_dim, 512, nn.SiLU(), NormPosition.POST),
            MLPLayer(512, 512, nn.SiLU(), NormPosition.POST),
            MLPLayer(512, 512, nn.Identity()),
        )

        for block_idx in range(self.num_blocks):
            setattr(self, f"block_{block_idx + 1}", AdaINResBlock(512, 512))

        self.head = GaussianHead(512, action_dim)

    def _iter_blocks(self):
        for block_idx in range(self.num_blocks):
            yield getattr(self, f"block_{block_idx + 1}")

    def forward(self, obs_dict:dict[str, torch.Tensor], action:torch.Tensor|None=None, update_normlizer:bool=False) -> StochasticContinuousPolicyStep:
        robot_obs = obs_dict["robot_obs"]
        motion_obs = obs_dict["motion_obs"]

        robot_obs = self.robot_obs_normlizer(robot_obs, update_normlizer)
        motion_obs = self.motion_obs_normlizer(motion_obs, update_normlizer)

        x_robot = self.robot_encoder(robot_obs)
        x_motion = self.motion_encoder(motion_obs)

        x = x_robot
        for block in self._iter_blocks():
            x = block(x, x_motion)
        
        step = self.head(x, action)

        return step
