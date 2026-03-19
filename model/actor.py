import torch
import torch.nn as nn

from RLAlg.normalizer import Normalizer
from RLAlg.nn.layers import MLPLayer, GaussianHead, NormPosition
from RLAlg.nn.steps import StochasticContinuousPolicyStep

from .adain import AdaINBlock, AdaINResBlock

class VanilaActor(nn.Module):
    def __init__(self, obs_dim:int, action_dim:int):
        super().__init__()

        self.normlizer = Normalizer((obs_dim,))

        self.encoder = nn.Sequential(
            MLPLayer(obs_dim, 512, nn.SiLU(), NormPosition.POST),
            MLPLayer(512, 512, nn.SiLU(), NormPosition.POST),
            MLPLayer(512, 512, nn.SiLU(), NormPosition.POST),
            MLPLayer(512, 512, nn.SiLU(), NormPosition.POST),
        )

        self.head = GaussianHead(512, action_dim)

    def forward(self, obs_dict:dict[str, torch.Tensor], action:torch.Tensor|None=None, update_normlizer:bool=False) -> StochasticContinuousPolicyStep:
        obs = obs_dict["obs"]
        obs = self.normlizer(obs, update=update_normlizer)
        x = self.encoder(obs)
        step = self.head(x, action)

        return step
    
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
        num_blocks: int = 5,
    ):
        super().__init__()

        if num_blocks < 1:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}.")

        self.robot_obs_normlizer = Normalizer((robot_obs_dim,))
        self.motion_obs_normlizer = Normalizer((motion_obs_dim,))
        self.num_blocks = num_blocks

        self.robot_encoder = nn.Sequential(
            MLPLayer(robot_obs_dim, 256, nn.SiLU(), NormPosition.POST),
            MLPLayer(256, 256, nn.SiLU(), NormPosition.POST),
            MLPLayer(256, 256, nn.Identity()),
        )

        self.motion_encoder = nn.Sequential(
            MLPLayer(motion_obs_dim, 256, nn.SiLU(), NormPosition.POST),
            MLPLayer(256, 256, nn.SiLU(), NormPosition.POST),
            MLPLayer(256, 256, nn.Identity()),
        )

        for block_idx in range(self.num_blocks):
            setattr(self, f"block_{block_idx + 1}", AdaINResBlock(256, 256))

        self.head = GaussianHead(256, action_dim)

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
