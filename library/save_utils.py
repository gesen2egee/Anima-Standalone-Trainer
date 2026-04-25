import json
import os
import shutil
import time
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from types import MethodType
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedType
from accelerate.utils.other import clean_state_dict_for_safetensors
from safetensors.torch import load_file as safetensors_load_file
from safetensors.torch import save_file as safetensors_save_file


SAVE_INTENT_LORA_STATE = "lora_state"
SAVE_INTENT_FULL_MODEL_EXPORT = "full_model_export"
SAVE_INTENT_RESUME_STATE_MODEL_PAYLOAD = "resume_state_model_payload"
PARALLEL_STATE_FORMAT_VERSION = 1


@dataclass(frozen=True)
class StateDictModelSpec:
    model: Any
    filename: str = "model"
    save_intent: str = SAVE_INTENT_FULL_MODEL_EXPORT
    unwrap_model: bool = True
    keep_torch_compile: bool = False
    target_model: Any = None


def is_fsdp_active(accelerator: Optional[Accelerator]) -> bool:
    return accelerator is not None and accelerator.distributed_type == DistributedType.FSDP


def unwrap_model(accelerator: Optional[Accelerator], model: Any, *, unwrap: bool, keep_torch_compile: bool):
    if model is None or accelerator is None or not unwrap:
        return model
    return accelerator.unwrap_model(model, keep_torch_compile=keep_torch_compile)


def get_model_state_dict_for_save(
    accelerator: Optional[Accelerator],
    model: Any,
    save_intent: str,
    *,
    unwrap_model_for_non_fsdp: bool = True,
    keep_torch_compile: bool = False,
):
    if model is None:
        return None

    # LoRA network is never FSDP-wrapped (kept outside accelerator.prepare), so state_dict() is safe directly.
    if save_intent == SAVE_INTENT_LORA_STATE:
        target_model = unwrap_model(
            accelerator,
            model,
            unwrap=unwrap_model_for_non_fsdp,
            keep_torch_compile=keep_torch_compile,
        )
        return target_model.state_dict()

    if is_fsdp_active(accelerator):
        return accelerator.get_state_dict(model)

    target_model = unwrap_model(
        accelerator,
        model,
        unwrap=unwrap_model_for_non_fsdp,
        keep_torch_compile=keep_torch_compile,
    )
    return target_model.state_dict()


@contextmanager
def override_model_state_dict(model: Any, state_dict: Dict[str, torch.Tensor]):
    original_state_dict = model.state_dict
    model.state_dict = MethodType(lambda _self, *args, **kwargs: state_dict, model)
    try:
        yield model
    finally:
        model.state_dict = original_state_dict


@contextmanager
def override_model_state_dicts_for_save(
    accelerator: Optional[Accelerator],
    specs: Sequence[StateDictModelSpec],
):
    with ExitStack() as stack:
        for spec in specs:
            state_dict = get_model_state_dict_for_save(
                accelerator,
                spec.model,
                spec.save_intent,
                unwrap_model_for_non_fsdp=spec.unwrap_model,
                keep_torch_compile=spec.keep_torch_compile,
            )
            target_model = spec.target_model if spec.target_model is not None else spec.model
            if target_model is None or state_dict is None:
                continue
            stack.enter_context(override_model_state_dict(target_model, state_dict))
        yield


def save_state_dict_to_safetensors(
    accelerator: Optional[Accelerator],
    model: Any,
    save_path: str,
    save_intent: str,
    *,
    metadata: Optional[Dict[str, str]] = None,
    unwrap_model_for_non_fsdp: bool = True,
    keep_torch_compile: bool = False,
):
    state_dict = get_model_state_dict_for_save(
        accelerator,
        model,
        save_intent,
        unwrap_model_for_non_fsdp=unwrap_model_for_non_fsdp,
        keep_torch_compile=keep_torch_compile,
    )
    state_dict = clean_state_dict_for_safetensors(state_dict)
    safetensors_save_file(state_dict, save_path, metadata=metadata or {"format": "pt"})


def write_train_state_metadata(output_dir: str, epoch: int, step: int):
    train_state_file = os.path.join(output_dir, "train_state.json")
    with open(train_state_file, "w", encoding="utf-8") as f:
        json.dump({"current_epoch": epoch, "current_step": step}, f)
    return train_state_file


def load_train_state_metadata(input_dir: str) -> Optional[Dict[str, int]]:
    train_state_file = os.path.join(input_dir, "train_state.json")
    if not os.path.exists(train_state_file):
        return None

    with open(train_state_file, "r", encoding="utf-8") as f:
        return json.load(f)


def parallel_rank_dir_name(rank: int) -> str:
    return f"rank_{rank:05d}"


def parallel_state_manifest_path(state_dir: str) -> str:
    return os.path.join(state_dir, "tp_state.json")


def parallel_state_rank_dir(state_dir: str, rank: int) -> str:
    return os.path.join(state_dir, parallel_rank_dir_name(rank))


def write_parallel_state_manifest(
    state_dir: str,
    *,
    parallel_size: int,
    backend: str,
    mode: str = "tp_sp",
) -> None:
    manifest = {
        "format_version": PARALLEL_STATE_FORMAT_VERSION,
        "mode": mode,
        "tp_size": int(parallel_size),
        "parallel_size": int(parallel_size),
        "backend": str(backend),
        "rank_dir_format": "rank_{rank:05d}",
    }
    with open(parallel_state_manifest_path(state_dir), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def read_parallel_state_manifest(state_dir: str) -> Optional[Dict[str, Any]]:
    manifest_path = parallel_state_manifest_path(state_dir)
    if not os.path.exists(manifest_path):
        return None
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_parallel_state_manifest(state_dir: str, *, parallel_size: int) -> Dict[str, Any]:
    manifest = read_parallel_state_manifest(state_dir)
    if manifest is None:
        raise FileNotFoundError(f"Parallel state manifest not found: {parallel_state_manifest_path(state_dir)}")

    saved_size = int(manifest.get("parallel_size", manifest.get("tp_size", -1)))
    if saved_size != int(parallel_size):
        raise ValueError(
            f"Parallel state was saved with parallel_size={saved_size}, "
            f"but current parallel_size={parallel_size}. Exact parallel resume "
            "requires the same degree."
        )
    return manifest


def install_parallel_state_wrappers(
    *,
    train_util_module,
    huggingface_util_module,
    parallel_rank: int,
    parallel_size: int,
    process_group=None,
    backend: str = "unknown",
    logger=None,
    mode: str = "tp_sp",
    patch_train_network_upload: bool = True,
) -> None:
    """Patch train_util save/resume helpers for rank-sharded parallel state folders.

    This preserves regular training's public state-folder names while saving one
    Accelerate state subfolder per tensor/sequence-parallel rank.
    """
    if getattr(train_util_module, "_parallel_state_wrappers_installed", False):
        return

    original_resume = train_util_module.resume_from_local_or_hf_if_specified
    original_hf_upload = huggingface_util_module.upload
    rank = int(parallel_rank)
    size = int(parallel_size)
    wait_timeout = float(os.environ.get("PARALLEL_STATE_WAIT_TIMEOUT_SEC", "1800"))

    def _log(message: str) -> None:
        if logger is not None:
            logger.info(message)

    def _write_marker(path: str, value: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_path = f"{path}.tmp.{os.getpid()}"
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(value)
        os.replace(tmp_path, path)

    def _wait_for_marker(path: str, *, expected_value: Optional[str] = None, label: str) -> str:
        deadline = time.monotonic() + wait_timeout
        last_value = None
        while time.monotonic() < deadline:
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        value = f.read().strip()
                except OSError:
                    value = None
                if expected_value is None or value == expected_value:
                    return value or ""
                last_value = value
            time.sleep(0.25)
        expected = "" if expected_value is None else f" with value {expected_value!r}"
        raise TimeoutError(
            f"Timed out after {wait_timeout:.0f}s waiting for parallel state {label}: "
            f"{path}{expected}. Last value: {last_value!r}"
        )

    def _save_id_path(state_dir: str) -> str:
        return os.path.join(state_dir, ".parallel_save_id")

    def _rank_done_path(state_dir: str, marker_rank: int) -> str:
        return os.path.join(state_dir, f".parallel_rank_{marker_rank:05d}.done")

    def _complete_path(state_dir: str) -> str:
        return os.path.join(state_dir, ".parallel_save_complete")

    def _wait_for_all_rank_markers(state_dir: str, save_id: str) -> None:
        for marker_rank in range(size):
            _wait_for_marker(
                _rank_done_path(state_dir, marker_rank),
                expected_value=save_id,
                label=f"rank {marker_rank} save completion",
            )

    def _upload_root_if_requested(args, state_dir: str, remote_name: str) -> None:
        if not getattr(args, "save_state_to_huggingface", False):
            return
        if rank != 0:
            return
        _log("uploading parallel state to huggingface.")
        original_hf_upload(args, state_dir, "/" + remote_name)

    def _copy_rank0_train_state_to_root(state_dir: str) -> None:
        metadata = load_train_state_metadata(parallel_state_rank_dir(state_dir, 0))
        if metadata is None:
            return
        write_train_state_metadata(
            state_dir,
            int(metadata.get("current_epoch", 0)),
            int(metadata.get("current_step", 0)),
        )

    def _save_parallel_state(args, accelerator, state_dir: str, remote_name: str) -> None:
        if rank == 0:
            os.makedirs(state_dir, exist_ok=True)
            write_parallel_state_manifest(
                state_dir,
                parallel_size=size,
                backend=backend,
                mode=mode,
            )
            save_id = str(time.time_ns())
            _write_marker(_save_id_path(state_dir), save_id)

        save_id = _wait_for_marker(_save_id_path(state_dir), label="save id")
        rank_dir = parallel_state_rank_dir(state_dir, rank)
        _log(f"[parallel state rank {rank}] saving rank state: {rank_dir}")
        accelerator.save_state(rank_dir)
        _write_marker(_rank_done_path(state_dir, rank), save_id)
        _wait_for_all_rank_markers(state_dir, save_id)

        if rank == 0:
            _copy_rank0_train_state_to_root(state_dir)
            _upload_root_if_requested(args, state_dir, remote_name)
            _write_marker(_complete_path(state_dir), save_id)
        _wait_for_marker(_complete_path(state_dir), expected_value=save_id, label="root completion")

    def _remove_state_dir_on_rank0(state_dir_old: str) -> None:
        if rank == 0 and os.path.exists(state_dir_old):
            _log(f"removing old parallel state: {state_dir_old}")
            shutil.rmtree(state_dir_old)

    def save_and_remove_state_on_epoch_end(args, accelerator, epoch_no):
        model_name = train_util_module.default_if_none(args.output_name, train_util_module.DEFAULT_EPOCH_NAME)

        _log("")
        _log(f"saving parallel state at epoch {epoch_no}")

        remote_name = train_util_module.EPOCH_STATE_NAME.format(model_name, epoch_no)
        state_dir = os.path.join(args.output_dir, remote_name)
        _save_parallel_state(args, accelerator, state_dir, remote_name)

        last_n_epochs = args.save_last_n_epochs_state if args.save_last_n_epochs_state else args.save_last_n_epochs
        if last_n_epochs is not None:
            remove_epoch_no = epoch_no - args.save_every_n_epochs * last_n_epochs
            state_dir_old = os.path.join(
                args.output_dir,
                train_util_module.EPOCH_STATE_NAME.format(model_name, remove_epoch_no),
            )
            _remove_state_dir_on_rank0(state_dir_old)

    def save_and_remove_state_stepwise(args, accelerator, step_no):
        model_name = train_util_module.default_if_none(args.output_name, train_util_module.DEFAULT_STEP_NAME)

        _log("")
        _log(f"saving parallel state at step {step_no}")

        remote_name = train_util_module.STEP_STATE_NAME.format(model_name, step_no)
        state_dir = os.path.join(args.output_dir, remote_name)
        _save_parallel_state(args, accelerator, state_dir, remote_name)

        last_n_steps = args.save_last_n_steps_state if args.save_last_n_steps_state else args.save_last_n_steps
        if last_n_steps is not None:
            remove_step_no = step_no - last_n_steps - 1
            remove_step_no = remove_step_no - (remove_step_no % args.save_every_n_steps)
            if remove_step_no > 0:
                state_dir_old = os.path.join(
                    args.output_dir,
                    train_util_module.STEP_STATE_NAME.format(model_name, remove_step_no),
                )
                _remove_state_dir_on_rank0(state_dir_old)

    def save_state_on_train_end(args, accelerator):
        model_name = train_util_module.default_if_none(args.output_name, train_util_module.DEFAULT_LAST_OUTPUT_NAME)

        _log("")
        _log("saving last parallel state.")

        remote_name = train_util_module.LAST_STATE_NAME.format(model_name)
        state_dir = os.path.join(args.output_dir, remote_name)
        _save_parallel_state(args, accelerator, state_dir, remote_name)

    def _download_hf_state_root(args) -> str:
        from huggingface_hub import hf_hub_download

        repo_id = args.resume.split("/")[0] + "/" + args.resume.split("/")[1]
        path_in_repo = "/".join(args.resume.split("/")[2:])
        revision = None
        repo_type = None
        if ":" in path_in_repo:
            divided = path_in_repo.split(":")
            if len(divided) == 2:
                path_in_repo, revision = divided
                repo_type = "model"
            else:
                path_in_repo, revision, repo_type = divided

        _log(f"Downloading parallel state from huggingface: {repo_id}/{path_in_repo}@{revision}")
        listed = huggingface_util_module.list_dir(
            repo_id=repo_id,
            subfolder=path_in_repo,
            revision=revision,
            token=args.huggingface_token,
            repo_type=repo_type,
        )
        if len(listed) == 0:
            raise ValueError("No files found in the specified Hugging Face state path.")

        downloaded = []
        for entry in listed:
            downloaded.append(
                hf_hub_download(
                    repo_id=repo_id,
                    filename=entry.rfilename,
                    revision=revision,
                    repo_type=repo_type,
                    token=args.huggingface_token,
                )
            )

        for path in downloaded:
            if os.path.basename(path) == "tp_state.json":
                return os.path.dirname(path)
        return os.path.dirname(downloaded[0])

    def resume_from_local_or_hf_if_specified(accelerator, args):
        if not args.resume:
            return

        if args.resume_from_huggingface:
            state_dir = _download_hf_state_root(args)
        else:
            state_dir = args.resume

        if read_parallel_state_manifest(state_dir) is None:
            return original_resume(accelerator, args)

        validate_parallel_state_manifest(state_dir, parallel_size=size)
        rank_dir = parallel_state_rank_dir(state_dir, rank)
        if not os.path.isdir(rank_dir):
            raise FileNotFoundError(f"Parallel state rank folder not found: {rank_dir}")

        _log(f"resume parallel training from local rank state: {rank_dir}")
        accelerator.load_state(rank_dir)

    def _rank0_upload_only(args, path, *upload_args, **upload_kwargs):
        if rank != 0:
            return None
        return original_hf_upload(args, path, *upload_args, **upload_kwargs)

    train_util_module.save_and_remove_state_on_epoch_end = save_and_remove_state_on_epoch_end
    train_util_module.save_and_remove_state_stepwise = save_and_remove_state_stepwise
    train_util_module.save_state_on_train_end = save_state_on_train_end
    train_util_module.resume_from_local_or_hf_if_specified = resume_from_local_or_hf_if_specified
    huggingface_util_module.upload = _rank0_upload_only
    train_util_module.huggingface_util.upload = _rank0_upload_only

    if patch_train_network_upload:
        try:
            import train_network
            train_network.huggingface_util.upload = _rank0_upload_only
        except Exception:
            pass

    train_util_module._parallel_state_wrappers_installed = True


def create_safetensors_state_hooks(
    accelerator: Accelerator,
    specs: Sequence[StateDictModelSpec],
    *,
    get_current_epoch: Callable[[], int],
    get_current_step: Callable[[], int],
    allow_non_main_process_save: bool = False,
    use_accelerate_native_fsdp: bool = False,
):
    state_tracker: Dict[str, Optional[int]] = {"current_step": None}

    def save_model_hook(models, weights, output_dir):
        native_fsdp = use_accelerate_native_fsdp and is_fsdp_active(accelerator)

        if not native_fsdp:
            if accelerator.is_main_process or allow_non_main_process_save:
                for spec in specs:
                    save_path = os.path.join(output_dir, f"{spec.filename}.safetensors")
                    save_state_dict_to_safetensors(
                        accelerator,
                        spec.model,
                        save_path,
                        spec.save_intent,
                        unwrap_model_for_non_fsdp=spec.unwrap_model,
                        keep_torch_compile=spec.keep_torch_compile,
                    )
            weights.clear()

        if accelerator.is_main_process:
            write_train_state_metadata(output_dir, get_current_epoch(), get_current_step())

    def load_model_hook(models, input_dir):
        metadata = load_train_state_metadata(input_dir)
        if metadata is not None:
            state_tracker["current_step"] = metadata["current_step"]

        native_fsdp = use_accelerate_native_fsdp and is_fsdp_active(accelerator)
        if native_fsdp:
            return

        for spec in specs:
            load_path = os.path.join(input_dir, f"{spec.filename}.safetensors")
            if not os.path.exists(load_path):
                continue

            base_model = unwrap_model(
                accelerator,
                spec.model,
                unwrap=spec.unwrap_model,
                keep_torch_compile=spec.keep_torch_compile,
            )
            state_dict = safetensors_load_file(load_path, device="cpu")
            base_model.load_state_dict(state_dict)

        models.clear()

    return save_model_hook, load_model_hook, state_tracker
