# region imports
import numpy as np
import torch
# endregion


class ReplayBuffer():
    def __init__(self, config, device="cuda") -> None:
        self.store_on_gpu = config.BasicSettings.ReplayBufferOnGPU
        max_length = config.JointTrainAgent.BufferMaxLength
        self.obs_mode = config.BasicSettings.get('ObsMode', 'features')
        if self.obs_mode != 'features':
            raise ValueError("FinMamba3 replay buffer only supports ObsMode='features'")
        obs_shape = (config.BasicSettings.FeatureDim,)
        obs_np_dtype = np.float32
        obs_torch_dtype = torch.float32
        self.device = device
        if self.store_on_gpu:
            self.obs_buffer = torch.empty((max_length, *obs_shape), dtype=obs_torch_dtype, device=device, requires_grad=False)
            self.action_buffer = torch.empty((max_length), dtype=torch.float32, device=device, requires_grad=False)
            self.reward_buffer = torch.empty((max_length), dtype=torch.float32, device=device, requires_grad=False)
            self.termination_buffer = torch.empty((max_length), dtype=torch.float32, device=device, requires_grad=False)
            # NaN sentinel: unresolved-market ticks must be visible to the loss path so BCE can mask them out instead of treating uninitialized memory as outcome=0.
            self.outcome_buffer = torch.full((max_length,), float('nan'), dtype=torch.float32, device=device, requires_grad=False)
            self.sampled_counter = torch.zeros((max_length), dtype=torch.int32, device=device, requires_grad=False)
            self.imagined_counter = torch.zeros((max_length), dtype=torch.int32, device=device, requires_grad=False)
            # Marks the final tick of each market. A dedicated buffer keeps the
            # sampling-boundary constraint separate from RL termination semantics.
            self.segment_end_buffer = torch.zeros((max_length), dtype=torch.float32, device=device, requires_grad=False)
        else:
            self.obs_buffer = np.empty((max_length, *obs_shape), dtype=obs_np_dtype)
            self.action_buffer = np.empty((max_length), dtype=np.float32)
            self.reward_buffer = np.empty((max_length), dtype=np.float32)
            self.termination_buffer = np.empty((max_length), dtype=np.float32)
            self.outcome_buffer = np.full((max_length,), np.nan, dtype=np.float32)
            self.sampled_counter = np.zeros((max_length), dtype=np.int32)
            self.imagined_counter = np.zeros((max_length), dtype=np.int32)
            self.segment_end_buffer = np.zeros((max_length), dtype=np.float32)
        self.length = 0
        self.last_pointer = -1
        self.max_length = max_length
        # Cache the boundary-aware start mask keyed by (length, batch_length). The
        # buffer is static after population, so each mask is built exactly once.
        self.valid_start_mask_by_key = {}
        self.world_model_warmup_length = config.JointTrainAgent.WorldModelWarmUp
        self.behaviour_warmup_length = config.JointTrainAgent.BehaviourWarmUp
        self.tau = config.JointTrainAgent.Tau
        self.imagination_tau = config.JointTrainAgent.ImaginationTau
        self.alpha = config.JointTrainAgent.Alpha
        self.beta = config.JointTrainAgent.Beta
        self.batch_scale_factor = config.JointTrainAgent.ImagineBatchSize / config.JointTrainAgent.BatchSize
    def ready(self, model_name='world_model'):
        return self.length  > self.world_model_warmup_length if model_name == 'world_model' else self.length  > self.behaviour_warmup_length
    def _valid_start_mask(self, batch_length):
        # A window [s, s + batch_length - 1] is valid iff no segment boundary falls
        # strictly inside it; a boundary at the final index is fine because the next
        # market starts past the window. With an all-zero segment buffer (single
        # market, FI-2010) every start is valid, so behavior matches the legacy path.
        key = (self.length, batch_length)
        cached_mask = self.valid_start_mask_by_key.get(key)
        if cached_mask is not None:
            return cached_mask
        num_starts = self.length + 1 - batch_length
        seg = self.segment_end_buffer[:self.length]
        if self.store_on_gpu:
            prefix = torch.cat([torch.zeros(1, device=self.device), torch.cumsum(seg.float(), dim=0)])
            crossings = prefix[batch_length - 1 : batch_length - 1 + num_starts] - prefix[:num_starts]
            mask = (crossings == 0).float()
        else:
            prefix = np.concatenate([[0.0], np.cumsum(seg.astype(np.float64))])
            crossings = prefix[batch_length - 1 : batch_length - 1 + num_starts] - prefix[:num_starts]
            mask = (crossings == 0).astype(np.float64)
        self.valid_start_mask_by_key[key] = mask
        return mask
    @torch.no_grad()
    def sample(self, batch_size, batch_length, imagine=False):
        if self.store_on_gpu:
            obs_list, action_list, reward_list, termination_list = [], [], [], []
            counts = self.sampled_counter[:self.length + 1 - batch_length]
            imagine_counts = self.imagined_counter[:self.length + 1 - batch_length] / self.batch_scale_factor
            # Sample with replacement when the batch exceeds the number of valid (boundary-free)
            # start positions, so a large batch over short markets cannot crash multinomial.
            valid_mask = self._valid_start_mask(batch_length)
            replacement = batch_size > int(valid_mask.sum().item())
            if imagine:
                linear_penalty = torch.maximum(torch.zeros_like(counts), counts - imagine_counts)
                score = counts - self.alpha * imagine_counts - self.beta * linear_penalty
                score = score / self.imagination_tau
                probabilities = torch.softmax(score, dim=0)
            else:
                # Numerically stable softmax over visit counts. The manual exp/sum form
                # underflows to all-zeros (then NaN) once counts grow large on big batches,
                # which crashes multinomial mid-training with a device-side assert.
                probabilities = torch.softmax(-counts / self.tau, dim=0)
            # Drop starts whose window would straddle a market boundary, then renormalize.
            probabilities = probabilities * valid_mask
            total_prob = probabilities.sum()
            if total_prob <= 0:
                raise RuntimeError("No valid boundary-free start positions for the requested batch_length")
            probabilities = probabilities / total_prob
            start_indexes = torch.multinomial(probabilities, batch_size, replacement=replacement)
            if not imagine:
                self.sampled_counter[start_indexes] += 1
            else:
                self.imagined_counter[start_indexes] += 1
            indexes = start_indexes.unsqueeze(-1).to(self.device) + torch.arange(batch_length, device=self.device)
            obs_list.append(self.obs_buffer[indexes])
            action_list.append(self.action_buffer[indexes])
            reward_list.append(self.reward_buffer[indexes])
            termination_list.append(self.termination_buffer[indexes])
            outcome_list = [self.outcome_buffer[indexes]]
            obs = torch.cat(obs_list, dim=0).float()
            action = torch.cat(action_list, dim=0)
            reward = torch.cat(reward_list, dim=0)
            termination = torch.cat(termination_list, dim=0)
            outcome = torch.cat(outcome_list, dim=0)
        else:
            obs_list, action_list, reward_list, termination_list = [], [], [], []
            outcome_list_cpu: list[torch.Tensor] = []
            if batch_size > 0:
                counts = self.sampled_counter[:self.length + 1 - batch_length]
                imagine_counts = self.imagined_counter[:self.length + 1 - batch_length] / self.batch_scale_factor
                valid_mask = self._valid_start_mask(batch_length)
                if imagine:
                    linear_penalty = np.maximum(np.zeros_like(counts), counts - imagine_counts)
                    score = counts - self.alpha * imagine_counts - self.beta * linear_penalty
                    score /= self.imagination_tau
                else:
                    score = -counts / self.tau
                exp_score = np.exp(score - np.max(score))
                probabilities = exp_score / np.sum(exp_score)
                # Drop starts whose window would straddle a market boundary, then renormalize.
                probabilities = probabilities * valid_mask
                total_prob = probabilities.sum()
                if total_prob <= 0:
                    raise RuntimeError("No valid boundary-free start positions for the requested batch_length")
                probabilities = probabilities / total_prob
                # Sample with replacement only when there are fewer valid starts than the batch size.
                replace = batch_size > int(valid_mask.sum())
                start_indexes = np.random.choice(len(probabilities), size=(batch_size,), replace=replace, p=probabilities)
                if not imagine:
                    self.sampled_counter[start_indexes] += 1
                else:
                    self.imagined_counter[start_indexes] += 1
                indexes = start_indexes[:, np.newaxis] + np.arange(batch_length)
                obs_seq = self.obs_buffer[indexes]
                action_seq = self.action_buffer[indexes]
                reward_seq = self.reward_buffer[indexes]
                termination_seq = self.termination_buffer[indexes]
                outcome_seq = self.outcome_buffer[indexes]
                obs_seq = torch.from_numpy(obs_seq).float().to(self.device)
                action_seq = torch.from_numpy(action_seq).to(self.device)
                reward_seq = torch.from_numpy(reward_seq).to(self.device)
                termination_seq = torch.from_numpy(termination_seq).to(self.device)
                outcome_seq = torch.from_numpy(outcome_seq).to(self.device)
                obs_list.append(obs_seq)
                action_list.append(action_seq)
                reward_list.append(reward_seq)
                termination_list.append(termination_seq)
                outcome_list_cpu.append(outcome_seq)
            obs = torch.cat(obs_list, dim=0) if obs_list else torch.empty(0, device=self.device)
            action = torch.cat(action_list, dim=0) if action_list else torch.empty(0, device=self.device)
            reward = torch.cat(reward_list, dim=0) if reward_list else torch.empty(0, device=self.device)
            termination = torch.cat(termination_list, dim=0) if termination_list else torch.empty(0, device=self.device)
            outcome = torch.cat(outcome_list_cpu, dim=0) if outcome_list_cpu else torch.empty(0, device=self.device)
        return obs, action, reward, termination, outcome
    def append(self, obs, action, reward, termination, outcome=float('nan'), segment_end=False):
        self.last_pointer = (self.last_pointer + 1) % (self.max_length)
        if self.store_on_gpu:
            self.obs_buffer[self.last_pointer] = torch.from_numpy(obs)
            self.action_buffer[self.last_pointer] = torch.tensor(action, device=self.device)
            self.reward_buffer[self.last_pointer] = torch.tensor(reward, device=self.device)
            self.termination_buffer[self.last_pointer] = torch.tensor(termination, device=self.device)
            self.outcome_buffer[self.last_pointer] = torch.tensor(float(outcome), device=self.device)
            self.segment_end_buffer[self.last_pointer] = float(segment_end)
        else:
            self.obs_buffer[self.last_pointer] = obs
            self.action_buffer[self.last_pointer] = action
            self.reward_buffer[self.last_pointer] = reward
            self.termination_buffer[self.last_pointer] = termination
            self.outcome_buffer[self.last_pointer] = float(outcome)
            self.segment_end_buffer[self.last_pointer] = float(segment_end)
        if len(self) < self.max_length:
            self.length += 1
    def __len__(self):
        return self.length
