from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Type

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import IGNORE_INDEX


class VRHMDTrajectoryDataset(Dataset):
    """
    OmniVLA-compatible dataset backed by JSONL manifests produced from Unity HMD recordings.
    """

    def __init__(
        self,
        action_tokenizer: ActionTokenizer,
        base_tokenizer: PreTrainedTokenizerBase,
        image_transform: ImageTransform,
        prompt_builder_fn: Type[PromptBuilder],
        manifest_path: str | Path,
        context_size: int,
        split: str = "train",
        modality_ids: tuple[int, ...] = (4, 5, 6, 7, 8),
        predict_stop_token: bool = True,
        image_size_history: tuple[int, int] = (96, 96),
        image_size_clip: tuple[int, int] = (224, 224),
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.split = split
        self.context_size = context_size
        self.modality_ids = modality_ids
        self.predict_stop_token = predict_stop_token
        self.image_size_history = image_size_history
        self.image_size_clip = image_size_clip

        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder = prompt_builder_fn

        self.samples = self._load_manifest(self.manifest_path)
        if not self.samples:
            raise RuntimeError(f"No samples found in manifest: {self.manifest_path}")

    def _load_manifest(self, manifest_path: Path) -> list[dict]:
        samples = []
        with manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _load_pil(self, path: str) -> Image.Image:
        return Image.open(path).convert("RGB")

    def _build_prompt_tensors(self, instruction: str, actions_np: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        current_action = torch.from_numpy(actions_np[0]).float()
        future_actions = torch.from_numpy(actions_np[1:]).float()
        future_actions_string = "".join(self.action_tokenizer(future_actions))
        current_action_string = self.action_tokenizer(current_action)
        action_chunk_string = current_action_string + future_actions_string
        action_chunk_len = len(action_chunk_string)

        prompt_builder = self.prompt_builder("openvla")
        conversation = [
            {"from": "human", "value": instruction},
            {"from": "gpt", "value": action_chunk_string},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        input_ids = torch.tensor(
            self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        )
        labels = input_ids.clone()
        labels[: -(action_chunk_len + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX
        return input_ids, labels

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        history_images = sample["history_images"]
        if len(history_images) != self.context_size + 1:
            raise ValueError(
                f"Expected {self.context_size + 1} history images, found {len(history_images)} in sample {idx}"
            )

        actions_np = np.asarray(sample["actions"], dtype=np.float32)
        goal_pose_np = np.asarray(sample["goal_pose"], dtype=np.float32)
        obj_pose_np = np.asarray(sample["obj_pose_norm"], dtype=np.float32)
        temp_dist_np = np.asarray(sample["temp_dist"], dtype=np.float32)

        modality_id = random.choice(self.modality_ids)
        use_language = modality_id in (7, 8)
        prompt_instruction = (
            f"What action should the robot take to {sample['instruction'].lower()}?"
            if use_language
            else "No language instruction"
        )
        input_ids, labels = self._build_prompt_tensors(prompt_instruction, actions_np)

        history_tensors = []
        for image_path in history_images:
            history_pil = self._load_pil(image_path)
            history_tensors.append(TF.resize(TF.to_tensor(history_pil), self.image_size_history))
        cur_image = torch.cat(history_tensors, dim=0)

        current_pil = self._load_pil(sample["current_image"]).resize(self.image_size_clip)
        goal_pil = self._load_pil(sample["goal_image"]).resize(self.image_size_clip)
        goal_image_8 = TF.resize(TF.to_tensor(goal_pil), self.image_size_history)

        if self.split == "train" and random.random() > 0.5:
            current_pil = current_pil.transpose(Image.FLIP_LEFT_RIGHT)
            goal_pil = goal_pil.transpose(Image.FLIP_LEFT_RIGHT)
            cur_image = torch.flip(cur_image, dims=[2])
            goal_image_8 = torch.flip(goal_image_8, dims=[2])
            actions_np = actions_np.copy()
            goal_pose_np = goal_pose_np.copy()
            obj_pose_np = obj_pose_np.copy()
            actions_np[:, 1] = -actions_np[:, 1]
            actions_np[:, 3] = -actions_np[:, 3]
            goal_pose_np[1] = -goal_pose_np[1]
            goal_pose_np[3] = -goal_pose_np[3]
            obj_pose_np[1] = -obj_pose_np[1]

        pixel_values = self.image_transform(current_pil)
        pixel_values_goal = self.image_transform(goal_pil)

        action_select_mask = np.asarray(1.0, dtype=np.float32)
        dataset_name = "vr_hmd"

        return dict(
            pixel_values=pixel_values,
            pixel_values_goal=pixel_values_goal,
            input_ids=input_ids,
            labels=labels,
            dataset_name=dataset_name,
            modality_id=modality_id,
            actions=actions_np,
            action_select_mask=action_select_mask,
            goal_pose=goal_pose_np,
            obj_pose_norm=obj_pose_np,
            img_PIL=current_pil,
            gimg_PIL=goal_pil,
            cur_image=cur_image.numpy(),
            goal_image_8=goal_image_8.numpy(),
            temp_dist=temp_dist_np,
            lan_prompt=sample["instruction"].lower(),
        )
