import transformers
import torch
from typing import *
import einops
from collections import UserDict
import copy

from src.training_types import *
from src.utilities import predict_observation


class BeamSearchScorer(transformers.BeamScorer):

    def __init__(
        self,
        cfg: Config,
        obs: torch.Tensor,
        batch_size: int,
        num_beams: int,
        device: torch.device,
        length_penalty: Optional[float] = 1.0,
        do_early_stopping: Optional[Union[bool, str]] = False,
        num_beam_hyps_to_keep: Optional[int] = 1,
        num_beam_groups: Optional[int] = 1,
        max_length: Optional[int] = None,
    ):
        self.cfg = cfg
        self.obs = obs
        self.num_beams = num_beams
        self.device = device
        self.length_penalty = length_penalty
        self.do_early_stopping = do_early_stopping
        self.num_beam_hyps_to_keep = num_beam_hyps_to_keep
        self.num_beam_groups = num_beam_groups
        self.group_size = self.num_beams // self.num_beam_groups

        self._beam_hyps = [0 for _ in range(self.cfg.batch_size)]
        self._is_done = False

    @property
    def is_done(self) -> bool:
        return self._is_done

    def process(
        self,
        input_ids: torch.LongTensor,
        next_scores: torch.FloatTensor,
        next_tokens: torch.LongTensor,
        next_indices: torch.LongTensor,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        beam_indices: Optional[torch.LongTensor] = None,
        group_index: Optional[int] = 0,
        decoder_prompt_len: Optional[int] = 0,
    ) -> Dict[str, torch.Tensor]:
        next_beam_scores = []
        next_beam_tokens = []
        next_beam_indices = []
        actions = input_ids[:, self.cfg.tok_p_action + self.cfg.tok_p_obs :]
        next_inds = copy.deepcopy(next_indices)
        for i in range(next_inds.shape[0]):
            next_inds[i] += self.cfg.num_beams * i
        next_inds = einops.rearrange(next_inds, "batch beam -> (batch beam)")
        next_tokens = einops.rearrange(next_tokens, "batch beam -> (batch beam)")
        new_actions = torch.cat((actions[next_inds], next_tokens.unsqueeze(1)), dim=-1)
        out = self.cfg.predictor_lm(new_actions, output_hidden_states=True)
        final_layer = out.hidden_states[-1]
        v_scores = self.cfg.inference_lm(final_layer)[:, -1, 0]
        # obs_losses, obs_tensor = predict_observation(
        #    self.cfg, new_actions, self.obs[next_inds]
        # )
        for i in range(self.cfg.batch_size):
            num_hypotheses = 2 * self.cfg.num_beams
            begin, end = i * num_hypotheses, (i + 1) * num_hypotheses
            # (-obs_losses)[begin:end] + next_scores[i]
            obs_scores = v_scores[begin:end]
            top_scores, top_indices = obs_scores.topk(k=self.cfg.num_beams)
            next_beam_scores.append(top_scores)
            next_beam_tokens.append(next_tokens[begin:end][top_indices])
            next_beam_indices.append(next_inds[begin:end][top_indices])
            if input_ids.shape[-1] == 2 * self.cfg.tok_p_action + self.cfg.tok_p_obs:
                self._is_done = True
            # print(f"Current generation index: {action.shape[-1]}")

        return UserDict(
            {
                "next_beam_scores": torch.cat(next_beam_scores),
                "next_beam_tokens": torch.cat(next_beam_tokens),
                "next_beam_indices": torch.cat(next_beam_indices),
            }
        )

        # selected_beams = self.beams[einops.rearrange(next_indices, "ba be -> (ba be)")]
        # augmented_beams = torch.cat(
        #    [
        #        selected_beams,
        #        einops.rearrange(next_tokens, "ba be -> (ba be)").unsqueeze(-1),
        #    ],
        #    dim=-1,
        # )
        # obs_losses, obs_tensor = predict_observation(self.cfg, augmented_beams, self.obs)
        # self.beam_scores, indices = torch.topk(
        #    obs_losses, k=self.cfg.batch_size * self.cfg.num_beams
        # )
        # self.beams = augmented_beams[indices]
        # return UserDict(
        #    {
        #        "next_beam_scores": self.beam_scores,
        #        "next_beam_tokens": self.beams[:, -1],
        #        "next_beam_indices": indices,
        #    }
        # )

    def finalize(
        self,
        input_ids: torch.LongTensor,
        final_beam_scores: torch.FloatTensor,
        final_beam_tokens: torch.LongTensor,
        final_beam_indices: torch.LongTensor,
        max_length: int,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        beam_indices: Optional[torch.LongTensor] = None,
        decoder_prompt_len: Optional[int] = 0,
    ) -> Tuple[torch.LongTensor]:
        batched_scores = einops.rearrange(
            final_beam_scores, "(batch beam) -> batch beam", batch=self.cfg.batch_size
        )
        top_scores, top_inds = batched_scores.topk(k=self.num_beam_hyps_to_keep, dim=1)
        all_inds = []
        for batch_index in range(self.cfg.batch_size):
            all_inds.append(top_inds[batch_index] + self.cfg.num_beams * batch_index)
        return UserDict({"sequences": input_ids[torch.cat(all_inds)]})
        # return 1
        # return UserDict(
        #    {
        #        "sequences": decoded,
        #        "sequence_scores": self.beam_scores,
        #        "beam_indices": self.beam_indices,
        #    }
        # )
