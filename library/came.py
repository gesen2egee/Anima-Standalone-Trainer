import math
from typing import Tuple

import torch


class NoSparseGradientError(Exception):
    def __init__(self, optimizer_name: str):
        super().__init__(f"{optimizer_name} does not support sparse gradient.")


class NoComplexParameterError(Exception):
    def __init__(self, optimizer_name: str):
        super().__init__(f"{optimizer_name} does not support complex parameter.")


class CAME(torch.optim.Optimizer):
    """Confidence-guided Adaptive Memory Efficient Optimization.

    Includes Automagic v3 dynamic per-row learning-rate control, an optional
    native Schedule-Free training path with weight decay applied on Z, and an
    optional Adapprox-style low-rank second-moment approximation path.

    Notes in this refactor:
    - Optimizer statistics are kept in a safe floating-point dtype to avoid
      fp16/bf16 underflow in the preconditioner.
    - Kahan compensation is kept in fp32-or-better and is not quantized.
    - Non-decoupled weight decay is applied before preconditioning, so it
      actually affects the current update.
    """

    _8BIT_STATE_NAMES = (
        "exp_avg",
        "exp_avg_sq",
        "exp_avg_sq_row",
        "exp_avg_sq_col",
        "exp_avg_res_row",
        "exp_avg_res_col",
        "exp_avg_sq_hat",
        "adapprox_exp_avg_sq_u",
        "adapprox_exp_avg_sq_s",
        "adapprox_exp_avg_sq_vh",
        "adapprox_exp_avg_res_u",
        "adapprox_exp_avg_res_s",
        "adapprox_exp_avg_res_vh",
    )

    def __init__(
        self,
        params,
        lr: float = 2e-4,
        betas: Tuple[float, float, float] = (0.9, 0.999, 0.9999),
        weight_decay: float = 0.1,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        clip_threshold: float = 1.0,
        ams_bound: bool = False,
        eps1: float = 1e-30,
        eps2: float = 1e-16,
        maximize: bool = False,
        automagic: bool = False,
        min_lr: float = 1e-7,
        max_lr: float = 1e-3,
        lr_bump_rate: float = 0.1,
        lr_smoothing_steps: int = 3,
        cautious_wd: bool = True,
        use_kahan: bool = True,
        use_magma: bool = True,
        mask_p: float = 0.5,
        magma_tau: float = 2.0,
        magma_ema: float = 0.9,
        mask_1d_params: bool = False,
        use_8bit: bool = False,
        min_8bit_size: int = 4096,
        schedulefree: bool = True,
        momentum: float = 0.9,
        r: float = 2.0,
        weight_lr_power: float = 2.0,
        Adapprox: bool = False,
        adapprox_rank: int = 8,
        adapprox_oversample: int = 4,
        adapprox_niter: int = 1,
        adapprox_residual: bool = True,
        **kwargs,
    ):
        # Accept both the requested public spelling `Adapprox=True` and the
        # conventional lowercase alias `adapprox=True`.
        if "adapprox" in kwargs:
            Adapprox = bool(kwargs.pop("adapprox"))

        if kwargs:
            unexpected = ", ".join(sorted(kwargs.keys()))
            raise TypeError(f"Unexpected optimizer arguments: {unexpected}")

        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, "weight_decay")
        self.validate_non_negative(eps1, "eps1")
        self.validate_non_negative(eps2, "eps2")

        if clip_threshold <= 0.0:
            raise ValueError(f"clip_threshold must be > 0. Got {clip_threshold}")
        if min_lr < 0.0:
            raise ValueError(f"min_lr must be >= 0. Got {min_lr}")
        if max_lr < 0.0:
            raise ValueError(f"max_lr must be >= 0. Got {max_lr}")
        if min_lr > max_lr:
            raise ValueError(f"min_lr must be <= max_lr. Got min_lr={min_lr}, max_lr={max_lr}")
        if lr_bump_rate < 0.0:
            raise ValueError(f"lr_bump_rate must be >= 0. Got {lr_bump_rate}")
        if lr_smoothing_steps < 1:
            raise ValueError(f"lr_smoothing_steps must be >= 1. Got {lr_smoothing_steps}")
        if not (0.0 < mask_p <= 1.0):
            raise ValueError(f"mask_p must be in (0, 1]. Got {mask_p}")
        if magma_tau <= 0.0:
            raise ValueError(f"magma_tau must be > 0. Got {magma_tau}")
        if not (0.0 <= magma_ema < 1.0):
            raise ValueError(f"magma_ema must be in [0, 1). Got {magma_ema}")
        if use_magma and betas[0] <= 0.0:
            raise ValueError("Magma requires beta1 > 0 because it uses first-moment momentum.")
        if min_8bit_size < 0:
            raise ValueError(f"min_8bit_size must be >= 0. Got {min_8bit_size}")
        if schedulefree and not (0.0 < momentum < 1.0):
            raise ValueError(f"momentum must be in (0, 1) when schedulefree=True. Got {momentum}")
        if r < 0.0:
            raise ValueError(f"r must be >= 0. Got {r}")
        if weight_lr_power < 0.0:
            raise ValueError(f"weight_lr_power must be >= 0. Got {weight_lr_power}")
        if adapprox_rank < 1:
            raise ValueError(f"adapprox_rank must be >= 1. Got {adapprox_rank}")
        if adapprox_oversample < 0:
            raise ValueError(f"adapprox_oversample must be >= 0. Got {adapprox_oversample}")
        if adapprox_niter < 0:
            raise ValueError(f"adapprox_niter must be >= 0. Got {adapprox_niter}")

        if automagic and lr > 1e-3:
            print("Warning! Start lr is very high for Automagic; forcing to 1e-6.")
            lr = 1e-6

        self.clip_threshold = float(clip_threshold)
        self.eps1 = float(eps1)
        self.eps2 = float(eps2)
        self.maximize = bool(maximize)
        self.schedulefree = bool(schedulefree)
        self.train_mode = False

        defaults = {
            "lr": float(lr),
            "betas": betas,
            "weight_decay": float(weight_decay),
            "weight_decouple": bool(weight_decouple),
            "fixed_decay": bool(fixed_decay),
            "ams_bound": bool(ams_bound),
            "eps1": float(eps1),
            "eps2": float(eps2),
            "automagic": bool(automagic),
            "min_lr": float(min_lr),
            "max_lr": float(max_lr),
            "lr_bump_rate": float(lr_bump_rate),
            "lr_smoothing_steps": int(lr_smoothing_steps),
            "dir_beta": float(lr_smoothing_steps) / (float(lr_smoothing_steps) + 1.0),
            "cautious_wd": bool(cautious_wd),
            "use_kahan": bool(use_kahan),
            "use_magma": bool(use_magma),
            "mask_p": float(mask_p),
            "magma_tau": float(magma_tau),
            "magma_ema": float(magma_ema),
            "mask_1d_params": bool(mask_1d_params),
            "use_8bit": bool(use_8bit),
            "min_8bit_size": int(min_8bit_size),
            "momentum": float(momentum),
            "r": float(r),
            "weight_lr_power": float(weight_lr_power),
            "Adapprox": bool(Adapprox),
            "adapprox_rank": int(adapprox_rank),
            "adapprox_oversample": int(adapprox_oversample),
            "adapprox_niter": int(adapprox_niter),
            "adapprox_residual": bool(adapprox_residual),
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return "CAME"

    @torch.no_grad()
    def eval(self):
        """Switch Schedule-Free parameters from extrapolated y to averaged x."""
        if self.schedulefree and self.train_mode:
            for group in self.param_groups:
                momentum = float(group["momentum"])
                for p in group["params"]:
                    state = self.state[p]
                    if "z" in state:
                        p.lerp_(end=state["z"], weight=1.0 - 1.0 / momentum)
            self.train_mode = False

    @torch.no_grad()
    def train(self):
        """Switch Schedule-Free parameters from averaged x to extrapolated y."""
        if self.schedulefree and not self.train_mode:
            for group in self.param_groups:
                momentum = float(group["momentum"])
                for p in group["params"]:
                    state = self.state[p]
                    if "z" in state:
                        p.lerp_(end=state["z"], weight=1.0 - momentum)
            self.train_mode = True

    @staticmethod
    def validate_non_negative(x, name: str) -> None:
        if x is not None and x < 0.0:
            raise ValueError(f"{name} must be non-negative")

    @staticmethod
    def validate_learning_rate(learning_rate) -> None:
        if learning_rate is not None and learning_rate < 0.0:
            raise ValueError(f"learning rate must be non-negative. Got {learning_rate}")

    @staticmethod
    def validate_range(x: float, name: str, low: float, high: float, range_type: str = "[)") -> None:
        if range_type == "[)" and not low <= x < high:
            raise ValueError(f"{name} must be in the range [{low}, {high})")
        if range_type == "[]" and not low <= x <= high:
            raise ValueError(f"{name} must be in the range [{low}, {high}]")

    def validate_betas(self, betas: Tuple[float, float, float]) -> None:
        if len(betas) != 3:
            raise ValueError(f"betas must contain 3 values. Got {betas}")
        self.validate_range(betas[0], "beta1", 0.0, 1.0, range_type="[)")
        self.validate_range(betas[1], "beta2", 0.0, 1.0, range_type="[)")
        self.validate_range(betas[2], "beta3", 0.0, 1.0, range_type="[]")

    @staticmethod
    def get_options(shape: Tuple[int, ...]) -> bool:
        return len(shape) >= 2

    @staticmethod
    def get_rms(x: torch.Tensor) -> torch.Tensor:
        return x.norm(2) / math.sqrt(x.numel())

    @staticmethod
    def _state_dtype_for(tensor: torch.Tensor) -> torch.dtype:
        if tensor.dtype in (torch.float16, torch.bfloat16):
            return torch.float32
        if tensor.dtype == torch.float64:
            return torch.float64
        return torch.float32

    @staticmethod
    def _tiny_for(dtype: torch.dtype) -> float:
        if dtype.is_floating_point:
            return float(torch.finfo(dtype).tiny)
        return float(torch.finfo(torch.float32).tiny)

    def _safe_eps(self, tensor: torch.Tensor, eps: float) -> float:
        dtype = tensor.dtype if tensor.is_floating_point() else torch.float32
        return max(float(eps), self._tiny_for(dtype))

    def init_group(self, group: dict, **kwargs) -> None:
        del kwargs

        group.setdefault("step", 0)
        group.setdefault("lr_max", 0.0)
        group.setdefault("weight_sum", 0.0)

        for p in group["params"]:
            if p.grad is None:
                continue

            grad = p.grad
            if grad.is_sparse:
                raise NoSparseGradientError(str(self))
            if torch.is_complex(p):
                raise NoComplexParameterError(str(self))

            state = self.state[p]
            if len(state) != 0:
                continue

            grad_shape: Tuple[int, ...] = grad.shape
            factored = self.get_options(grad_shape)
            state_dtype = self._state_dtype_for(p)

            if self.schedulefree:
                state["z"] = p.detach().clone(memory_format=torch.preserve_format)

            state["exp_avg"] = torch.zeros_like(p, dtype=state_dtype)

            if factored:
                if group["Adapprox"]:
                    # Low-rank factors are initialized lazily on the first step,
                    # because their rank depends on the flattened matrix size.
                    state["adapprox_exp_avg_sq_u"] = None
                    state["adapprox_exp_avg_sq_s"] = None
                    state["adapprox_exp_avg_sq_vh"] = None
                    if group["adapprox_residual"]:
                        state["adapprox_exp_avg_res_u"] = None
                        state["adapprox_exp_avg_res_s"] = None
                        state["adapprox_exp_avg_res_vh"] = None
                    else:
                        state["exp_avg_res_row"] = torch.zeros(grad_shape[:-1], dtype=state_dtype, device=grad.device)
                        state["exp_avg_res_col"] = torch.zeros(grad_shape[:-2] + grad_shape[-1:], dtype=state_dtype, device=grad.device)
                else:
                    state["exp_avg_sq_row"] = torch.zeros(grad_shape[:-1], dtype=state_dtype, device=grad.device)
                    state["exp_avg_sq_col"] = torch.zeros(grad_shape[:-2] + grad_shape[-1:], dtype=state_dtype, device=grad.device)
                    state["exp_avg_res_row"] = torch.zeros(grad_shape[:-1], dtype=state_dtype, device=grad.device)
                    state["exp_avg_res_col"] = torch.zeros(grad_shape[:-2] + grad_shape[-1:], dtype=state_dtype, device=grad.device)
            else:
                state["exp_avg_sq"] = torch.zeros_like(grad, dtype=state_dtype)

            if group["ams_bound"]:
                state["exp_avg_sq_hat"] = torch.zeros_like(grad, dtype=state_dtype)

            if group["automagic"]:
                initial_lr = max(float(group["min_lr"]), min(float(group["lr"]), float(group["max_lr"])))
                lr_shape = (p.shape[0],) if p.dim() >= 2 else p.shape
                state["layer_lr"] = torch.full(lr_shape, initial_lr, dtype=torch.float32, device=p.device)
                state["prev_sign"] = None
                state["dir_ema"] = torch.zeros(lr_shape, dtype=torch.float32, device=p.device)
                state["avg_lr"] = initial_lr
                state["lr_max_val"] = max(initial_lr, 1e-8)
            else:
                state["avg_lr"] = float(group["lr"])
                state["lr_max_val"] = max(float(group["lr"]), 1e-8)

            if group["use_kahan"]:
                state["kahan_comp"] = torch.zeros_like(p, dtype=state_dtype)

            if group["use_magma"]:
                state["magma_s"] = torch.tensor(1.0, device=p.device, dtype=torch.float32)

            state["RMS"] = 0.0

    @staticmethod
    def approximate_sq_grad(
        exp_avg_sq_row: torch.Tensor,
        exp_avg_sq_col: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        tiny = torch.finfo(output.dtype).tiny
        row_mean = exp_avg_sq_row.mean(dim=-1, keepdim=True).clamp_min(tiny)
        r_factor = (exp_avg_sq_row / row_mean).clamp_min(tiny).rsqrt_().unsqueeze(-1)
        c_factor = exp_avg_sq_col.clamp_min(tiny).rsqrt().unsqueeze(-2)
        torch.mul(r_factor, c_factor, out=output)

    @staticmethod
    def _flatten_to_matrix(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, ...]]:
        """Flatten a tensor to a 2D matrix, preserving the last dimension."""
        original_shape = tuple(x.shape)
        return x.reshape(-1, original_shape[-1]), original_shape

    @staticmethod
    def _unflatten_from_matrix(x: torch.Tensor, original_shape: Tuple[int, ...]) -> torch.Tensor:
        return x.reshape(original_shape)

    @staticmethod
    def _reconstruct_low_rank(
        state: dict,
        prefix: str,
        shape: Tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        u = state.get(f"{prefix}_u")
        s = state.get(f"{prefix}_s")
        vh = state.get(f"{prefix}_vh")

        if (
            u is None
            or s is None
            or vh is None
            or not torch.is_tensor(u)
            or not torch.is_tensor(s)
            or not torch.is_tensor(vh)
            or tuple(u.shape[:1]) != (shape[0],)
            or tuple(vh.shape[1:]) != (shape[1],)
        ):
            return torch.zeros(shape, device=device, dtype=dtype)

        return (u.to(dtype) * s.to(dtype).unsqueeze(0)) @ vh.to(dtype)

    @staticmethod
    def _randomized_low_rank_approx(
        matrix: torch.Tensor,
        rank: int,
        oversample: int,
        niter: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return low-rank approximation and factors via randomized SVD.

        This is an Adapprox-style practical path: optimizer state stores only
        U/S/Vh factors, while the full dense approximation exists only as a
        temporary tensor during the step. That is acceptable for LoRA-sized
        matrices and avoids changing the public optimizer API.
        """
        m, n = matrix.shape
        max_rank = min(m, n)
        rank = max(1, min(int(rank), max_rank))
        sketch_rank = max(rank, min(rank + int(oversample), max_rank))

        # Randomized range finder.
        omega = torch.randn(n, sketch_rank, device=matrix.device, dtype=matrix.dtype)
        y = matrix @ omega
        for _ in range(int(niter)):
            y = matrix @ (matrix.transpose(0, 1) @ y)

        q, _ = torch.linalg.qr(y, mode="reduced")
        b = q.transpose(0, 1) @ matrix

        u_hat, s, vh = torch.linalg.svd(b, full_matrices=False)
        u = q @ u_hat[:, :rank]
        s = s[:rank]
        vh = vh[:rank, :]

        approx = (u * s.unsqueeze(0)) @ vh
        return approx, u, s, vh

    @torch.no_grad()
    def _adapprox_preconditioner(
        self,
        sample: torch.Tensor,
        state: dict,
        group: dict,
        prefix: str,
        beta: float,
        eps: float,
    ) -> torch.Tensor:
        matrix, original_shape = self._flatten_to_matrix(sample)
        matrix = matrix.to(sample.dtype)

        prev = self._reconstruct_low_rank(
            state=state,
            prefix=prefix,
            shape=tuple(matrix.shape),
            device=matrix.device,
            dtype=matrix.dtype,
        )
        ema_matrix = prev.mul(beta).add_(matrix, alpha=1.0 - beta)

        approx, u, s, vh = self._randomized_low_rank_approx(
            matrix=ema_matrix,
            rank=int(group["adapprox_rank"]),
            oversample=int(group["adapprox_oversample"]),
            niter=int(group["adapprox_niter"]),
        )

        state[f"{prefix}_u"] = u.detach()
        state[f"{prefix}_s"] = s.detach()
        state[f"{prefix}_vh"] = vh.detach()

        # Randomized low-rank reconstruction can contain small negative values.
        preconditioner = approx.clamp_min(eps).rsqrt()
        return self._unflatten_from_matrix(preconditioner, original_shape)

    def _dequantize_state_tensors(self, state: dict) -> None:
        for name in self._8BIT_STATE_NAMES:
            tensor = state.get(name)
            scale = state.get(f"{name}_8bit_scale")
            if not torch.is_tensor(tensor) or tensor.dtype != torch.uint8 or scale is None:
                continue
            original_dtype = state.get(f"{name}_8bit_dtype", torch.float32)
            scale = scale.to(device=tensor.device, dtype=torch.float32)
            state[name] = (tensor.to(torch.float32) - 128.0).mul_(scale).to(original_dtype)

    def _quantize_state_tensors(self, state: dict, group: dict) -> None:
        if not group["use_8bit"]:
            return

        min_size = int(group["min_8bit_size"])
        for name in self._8BIT_STATE_NAMES:
            tensor = state.get(name)
            if (
                not torch.is_tensor(tensor)
                or tensor.dtype == torch.uint8
                or not tensor.is_floating_point()
                or tensor.numel() < min_size
            ):
                continue

            fp32_tensor = tensor.detach().to(torch.float32)
            max_abs = fp32_tensor.abs().max()
            if not bool(torch.isfinite(max_abs).item()) or float(max_abs.item()) == 0.0:
                scale = torch.ones((), device=tensor.device, dtype=torch.float32)
                quantized = torch.full(tensor.shape, 128, dtype=torch.uint8, device=tensor.device)
            else:
                scale = (max_abs / 127.0).to(torch.float32)
                quantized = torch.clamp(torch.round(fp32_tensor / scale) + 128.0, 0, 255).to(torch.uint8)

            state[name] = quantized
            state[f"{name}_8bit_scale"] = scale.detach()
            state[f"{name}_8bit_dtype"] = tensor.dtype

    def _effective_external_lr(self, group: dict) -> float:
        """Use the highest external LR seen so far while Schedule-Free is active.

        This prevents an outside scheduler from decaying the actual update LR
        after warmup, while still allowing warmup/increases to raise the lock.
        The raw group["lr"] is intentionally left untouched so external
        schedulers can keep their own state.
        """
        lr = float(group["lr"])
        if self.schedulefree:
            return max(lr, float(group.get("lr_max", lr)))
        return lr

    @torch.no_grad()
    def get_learning_rates(self):
        learning_rates = []
        for group in self.param_groups:
            group_rates = []
            for p in group["params"]:
                state = self.state.get(p)
                if state and "avg_lr" in state:
                    group_rates.append(float(state["avg_lr"]))
            learning_rates.append(sum(group_rates) / len(group_rates) if group_rates else float(group["lr"]))
        return learning_rates

    @torch.no_grad()
    def get_avg_learning_rate(self) -> float:
        learning_rates = self.get_learning_rates()
        return sum(learning_rates) / len(learning_rates) if learning_rates else 0.0

    @torch.no_grad()
    def get_lr_to_use(
        self,
        update: torch.Tensor,
        p: torch.Tensor,
        state: dict,
        group: dict,
    ) -> torch.Tensor:
        if group["automagic"]:
            layer_lr = state["layer_lr"]
            dir_beta = float(group["dir_beta"])
            lr_bump_rate = float(group["lr_bump_rate"])

            cur_sign = update.sign().to(torch.int8)
            prev_sign = state.get("prev_sign", None)

            if prev_sign is None:
                state["prev_sign"] = cur_sign.clone()
                lr_to_use = layer_lr
            else:
                flip_signal = cur_sign * prev_sign
                state["prev_sign"] = cur_sign.clone()
                vote_mask = flip_signal != 0

                if p.dim() >= 2:
                    reduce_dims = tuple(range(1, flip_signal.dim()))
                    sum_signal = torch.where(vote_mask, flip_signal.to(torch.float32), 0.0).sum(dim=reduce_dims)
                    sum_mask = vote_mask.to(torch.float32).sum(dim=reduce_dims).clamp_min(1.0)
                    row_vote = sum_signal / sum_mask
                else:
                    row_vote = torch.where(vote_mask, flip_signal.to(torch.float32), 0.0)

                dir_ema = state["dir_ema"]
                dir_ema.mul_(dir_beta).add_(row_vote, alpha=1.0 - dir_beta)

                layer_lr.mul_(torch.exp(dir_ema * lr_bump_rate)).clamp_(
                    min=float(group["min_lr"]),
                    max=float(group["max_lr"]),
                )
                lr_to_use = layer_lr

            state["avg_lr"] = float(lr_to_use.mean().detach().cpu().item())
        else:
            effective_lr = self._effective_external_lr(group)
            lr_to_use = torch.tensor(effective_lr, device=p.device, dtype=torch.float32)
            state["avg_lr"] = effective_lr

        current_avg_lr = float(lr_to_use.mean().detach().cpu().item())
        state["lr_max_val"] = max(current_avg_lr, float(state.get("lr_max_val", float(group["lr"]))))

        if p.dim() >= 2 and lr_to_use.numel() > 1:
            return lr_to_use.view(p.shape[0], *((1,) * (p.dim() - 1)))
        return lr_to_use

    @torch.no_grad()
    def apply_decoupled_weight_decay(
        self,
        target: torch.Tensor,
        reference: torch.Tensor,
        update: torch.Tensor,
        group: dict,
        state: dict,
        lr_to_use: torch.Tensor,
    ) -> None:
        weight_decay = float(group["weight_decay"])
        if weight_decay == 0.0 or not group["weight_decouple"]:
            return

        if group["fixed_decay"]:
            decay_scale = torch.tensor(weight_decay, device=target.device, dtype=torch.float32)
        else:
            base_lr = self._effective_external_lr(group)
            lr_max_val = max(float(state.get("lr_max_val", base_lr)), 1e-8)
            decay_scale = weight_decay * lr_to_use * (base_lr / lr_max_val)

        decay_scale = decay_scale.to(torch.float32)
        target_fp32 = target.detach().to(torch.float32)

        if group["cautious_wd"]:
            decay_mask = (update.to(torch.float32) * reference.detach().to(torch.float32) >= 0).to(torch.float32)
            decayed = target_fp32 - decay_scale * target_fp32 * decay_mask
        else:
            decayed = target_fp32 - decay_scale * target_fp32

        target.copy_(decayed.to(target.dtype))

    @torch.no_grad()
    def apply_magma(
        self,
        update_tensor: torch.Tensor,
        alignment_tensor: torch.Tensor,
        first_moment: torch.Tensor,
        state: dict,
        group: dict,
        p: torch.Tensor,
    ) -> torch.Tensor:
        if not group["use_magma"]:
            return update_tensor
        if update_tensor.ndim < 2 and not group["mask_1d_params"]:
            return update_tensor

        mask_p = float(group["mask_p"])
        tau = float(group["magma_tau"])
        ema = float(group["magma_ema"])

        momentum_flat = first_moment.reshape(-1).float()
        align_flat = alignment_tensor.reshape(-1).float()

        denom = (momentum_flat.norm(p=2) * align_flat.norm(p=2)).clamp_min(1e-12)
        cosine_similarity = (momentum_flat.dot(align_flat) / denom).clamp(-1.0, 1.0)

        s_hat = torch.sigmoid(cosine_similarity / tau).to(torch.float32)
        prev_score = state.get("magma_s")
        if prev_score is None or not torch.is_tensor(prev_score) or prev_score.device != p.device:
            prev_score = torch.tensor(1.0, device=p.device, dtype=torch.float32)

        score = ema * prev_score + (1.0 - ema) * s_hat
        state["magma_s"] = score.detach()

        mask = (torch.rand((), device=p.device) < mask_p).to(update_tensor.dtype)
        return update_tensor * mask * score.to(update_tensor.dtype)

    @torch.no_grad()
    def apply_kahan_update(
        self,
        target_tensor: torch.Tensor,
        update_tensor: torch.Tensor,
        state: dict,
    ) -> None:
        kahan_comp = state["kahan_comp"]
        target_fp = target_tensor.detach().to(kahan_comp.dtype)
        value_to_add = -update_tensor.to(kahan_comp.dtype)
        compensated_update = value_to_add - kahan_comp

        new_target_fp = target_fp + compensated_update
        new_target = new_target_fp.to(target_tensor.dtype)
        actual_delta = new_target.to(kahan_comp.dtype) - target_fp
        new_comp = actual_delta - compensated_update

        kahan_comp.copy_(new_comp)
        target_tensor.copy_(new_target)

    def _make_grad_for_update(self, grad: torch.Tensor, decay_reference: torch.Tensor, group: dict) -> torch.Tensor:
        state_dtype = self._state_dtype_for(decay_reference)
        grad_work = grad.detach().to(state_dtype)
        if self.maximize:
            grad_work = -grad_work

        if not group["weight_decouple"] and float(group["weight_decay"]) > 0.0:
            grad_work = grad_work + decay_reference.detach().to(state_dtype) * float(group["weight_decay"])

        return grad_work

    def _precondition_gradient(self, grad_work: torch.Tensor, state: dict, group: dict, factored: bool) -> torch.Tensor:
        beta1, beta2, beta3 = group["betas"]
        eps1 = self._safe_eps(grad_work, self.eps1)
        eps2 = self._safe_eps(grad_work, self.eps2)

        sq_grad = grad_work.square().add_(eps1)

        if factored:
            if group["Adapprox"]:
                preconditioner = self._adapprox_preconditioner(
                    sample=sq_grad,
                    state=state,
                    group=group,
                    prefix="adapprox_exp_avg_sq",
                    beta=beta2,
                    eps=eps1,
                )
            else:
                exp_avg_sq_row = state["exp_avg_sq_row"]
                exp_avg_sq_col = state["exp_avg_sq_col"]
                exp_avg_sq_row.mul_(beta2).add_(sq_grad.mean(dim=-1), alpha=1.0 - beta2)
                exp_avg_sq_col.mul_(beta2).add_(sq_grad.mean(dim=-2), alpha=1.0 - beta2)
                preconditioner = torch.empty_like(grad_work)
                self.approximate_sq_grad(exp_avg_sq_row, exp_avg_sq_col, preconditioner)
        else:
            exp_avg_sq = state["exp_avg_sq"]
            exp_avg_sq.mul_(beta2).add_(sq_grad, alpha=1.0 - beta2)
            preconditioner = exp_avg_sq.clamp_min(eps1).rsqrt()

        if group["ams_bound"]:
            exp_avg_sq_hat = state["exp_avg_sq_hat"]
            estimated_second_moment = preconditioner.clamp_min(eps1).reciprocal().square()
            torch.maximum(exp_avg_sq_hat, estimated_second_moment, out=exp_avg_sq_hat)
            preconditioner = exp_avg_sq_hat.clamp_min(eps1).rsqrt()

        update = preconditioner.mul(grad_work)
        update.div_((self.get_rms(update) / self.clip_threshold).clamp(min=1.0))

        exp_avg = state["exp_avg"]
        exp_avg.mul_(beta1).add_(update, alpha=1.0 - beta1)

        if not factored:
            return exp_avg

        residual = update.sub(exp_avg).square_().add_(eps2)

        if group["Adapprox"] and group["adapprox_residual"]:
            residual_preconditioner = self._adapprox_preconditioner(
                sample=residual,
                state=state,
                group=group,
                prefix="adapprox_exp_avg_res",
                beta=beta3,
                eps=eps2,
            )
        else:
            exp_avg_res_row = state["exp_avg_res_row"]
            exp_avg_res_col = state["exp_avg_res_col"]
            exp_avg_res_row.mul_(beta3).add_(residual.mean(dim=-1), alpha=1.0 - beta3)
            exp_avg_res_col.mul_(beta3).add_(residual.mean(dim=-2), alpha=1.0 - beta3)

            residual_preconditioner = torch.empty_like(update)
            self.approximate_sq_grad(exp_avg_res_row, exp_avg_res_col, residual_preconditioner)

        return residual_preconditioner.mul_(exp_avg)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self.init_group(group)

            if self.schedulefree:
                if not self.train_mode:
                    raise RuntimeError(
                        "Optimizer is not in train mode. Call optimizer.train() before the training loop."
                    )

                k = int(group["step"])
                # Lock to the maximum external LR observed so far. This keeps
                # Schedule-Free from silently inheriting a cosine/linear decay
                # after warmup, without mutating group["lr"] itself.
                lr_max = group["lr_max"] = max(float(group["lr"]), float(group.get("lr_max", 0.0)))
                weight = ((k + 1) ** float(group["r"])) * (lr_max ** float(group["weight_lr_power"]))
                group["weight_sum"] = float(group.get("weight_sum", 0.0)) + float(weight)
                ckp1 = 0.0 if group["weight_sum"] == 0.0 else float(weight) / float(group["weight_sum"])

            group["step"] += 1

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                self._dequantize_state_tensors(state)

                grad = p.grad
                target = state["z"] if self.schedulefree else p
                factored = self.get_options(tuple(grad.shape))

                state["RMS"] = float(self.get_rms(target.detach().float()).detach().cpu().item())

                grad_work = self._make_grad_for_update(grad, target, group)
                update = self._precondition_gradient(grad_work, state, group, factored)

                lr_to_use = self.get_lr_to_use(update=update, p=target, state=state, group=group)

                self.apply_decoupled_weight_decay(
                    target=target,
                    reference=p if self.schedulefree else p,
                    update=update,
                    group=group,
                    state=state,
                    lr_to_use=lr_to_use,
                )

                update_tensor = update * lr_to_use.to(update.dtype)

                if group["use_magma"]:
                    update_tensor = self.apply_magma(
                        update_tensor=update_tensor,
                        alignment_tensor=update,
                        first_moment=state["exp_avg"],
                        state=state,
                        group=group,
                        p=target,
                    )

                if self.schedulefree:
                    momentum = float(group["momentum"])

                    # y -> x
                    p.lerp_(end=target, weight=1.0 - 1.0 / momentum)

                    # z update
                    if group["use_kahan"]:
                        self.apply_kahan_update(target_tensor=target, update_tensor=update_tensor, state=state)
                    else:
                        target.add_(-update_tensor.to(target.dtype))

                    # x update
                    p.lerp_(end=target, weight=ckp1)

                    # x -> next y
                    p.lerp_(end=target, weight=1.0 - momentum)
                else:
                    if group["use_kahan"]:
                        self.apply_kahan_update(target_tensor=p, update_tensor=update_tensor, state=state)
                    else:
                        p.add_(-update_tensor.to(p.dtype))

                self._quantize_state_tensors(state, group)

        return loss
