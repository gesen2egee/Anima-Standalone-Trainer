import importlib.machinery
import os
import random
import sys
import types
import unittest


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if "xformers" not in sys.modules:
    xformers_module = types.ModuleType("xformers")
    xformers_ops_module = types.ModuleType("xformers.ops")
    xformers_module.__spec__ = importlib.machinery.ModuleSpec("xformers", loader=None)
    xformers_ops_module.__spec__ = importlib.machinery.ModuleSpec("xformers.ops", loader=None)
    xformers_module.ops = xformers_ops_module
    sys.modules["xformers"] = xformers_module
    sys.modules["xformers.ops"] = xformers_ops_module

if "diffusers" not in sys.modules:
    diffusers_module = types.ModuleType("diffusers")
    diffusers_module.__spec__ = importlib.machinery.ModuleSpec("diffusers", loader=None)

    class _DummyDiffusersClass:
        pass

    for class_name in [
        "StableDiffusionPipeline",
        "DDPMScheduler",
        "EulerAncestralDiscreteScheduler",
        "DPMSolverMultistepScheduler",
        "DPMSolverSinglestepScheduler",
        "LMSDiscreteScheduler",
        "PNDMScheduler",
        "DDIMScheduler",
        "EulerDiscreteScheduler",
        "HeunDiscreteScheduler",
        "KDPM2DiscreteScheduler",
        "KDPM2AncestralDiscreteScheduler",
        "AutoencoderKL",
        "UNet2DConditionModel",
        "SchedulerMixin",
    ]:
        setattr(diffusers_module, class_name, _DummyDiffusersClass)

    diffusers_optimization_module = types.ModuleType("diffusers.optimization")
    diffusers_optimization_module.__spec__ = importlib.machinery.ModuleSpec("diffusers.optimization", loader=None)
    diffusers_optimization_module.SchedulerType = _DummyDiffusersClass
    diffusers_optimization_module.TYPE_TO_SCHEDULER_FUNCTION = {}
    diffusers_models_module = types.ModuleType("diffusers.models")
    diffusers_models_module.__spec__ = importlib.machinery.ModuleSpec("diffusers.models", loader=None)
    diffusers_models_module.AutoencoderKL = _DummyDiffusersClass
    diffusers_models_module.UNet2DConditionModel = _DummyDiffusersClass
    diffusers_schedulers_module = types.ModuleType("diffusers.schedulers")
    diffusers_schedulers_module.__spec__ = importlib.machinery.ModuleSpec("diffusers.schedulers", loader=None)
    diffusers_euler_module = types.ModuleType("diffusers.schedulers.scheduling_euler_ancestral_discrete")
    diffusers_euler_module.__spec__ = importlib.machinery.ModuleSpec(
        "diffusers.schedulers.scheduling_euler_ancestral_discrete", loader=None
    )
    diffusers_euler_module.EulerAncestralDiscreteSchedulerOutput = _DummyDiffusersClass
    diffusers_ddpm_module = types.ModuleType("diffusers.schedulers.scheduling_ddpm")
    diffusers_ddpm_module.__spec__ = importlib.machinery.ModuleSpec("diffusers.schedulers.scheduling_ddpm", loader=None)
    diffusers_ddpm_module.DDPMScheduler = _DummyDiffusersClass

    sys.modules["diffusers"] = diffusers_module
    sys.modules["diffusers.optimization"] = diffusers_optimization_module
    sys.modules["diffusers.models"] = diffusers_models_module
    sys.modules["diffusers.schedulers"] = diffusers_schedulers_module
    sys.modules["diffusers.schedulers.scheduling_euler_ancestral_discrete"] = diffusers_euler_module
    sys.modules["diffusers.schedulers.scheduling_ddpm"] = diffusers_ddpm_module

for module_name, attributes in {
    "library.custom_train_functions": {},
    "library.sd3_utils": {},
    "library.model_util": {},
    "library.huggingface_util": {},
    "library.sai_model_spec": {"ModelSpecMetadata": object},
    "library.deepspeed_utils": {},
    "library.lpw_stable_diffusion": {"StableDiffusionLongPromptWeightingPipeline": object},
    "library.sdxl_lpw_stable_diffusion": {"SdxlStableDiffusionLongPromptWeightingPipeline": object},
}.items():
    if module_name not in sys.modules:
        module = types.ModuleType(module_name)
        module.__spec__ = importlib.machinery.ModuleSpec(module_name, loader=None)
        for attr_name, attr_value in attributes.items():
            setattr(module, attr_name, attr_value)
        sys.modules[module_name] = module

from library.train_util import BaseDataset, BaseSubset, ControlNetDataset, DreamBoothDataset, FineTuningDataset, ImageInfo


def create_subset(enable_wildcard=False, enable_fad=False, fad_curriculum=False, keep_tags=None):
    return BaseSubset(
        image_dir="/mock/img/dir",
        alpha_mask=False,
        num_repeats=1,
        shuffle_caption=False,
        caption_separator=",",
        keep_tokens=0,
        keep_tokens_separator=None,
        secondary_separator=None,
        enable_wildcard=enable_wildcard,
        color_aug=False,
        flip_aug=False,
        face_crop_aug_range=None,
        random_crop=False,
        caption_dropout_rate=0.0,
        caption_dropout_every_n_epochs=0,
        caption_tag_dropout_rate=0.0,
        caption_prefix=None,
        caption_suffix=None,
        token_warmup_min=0,
        token_warmup_step=0,
        enable_fad=enable_fad,
        fad_curriculum=fad_curriculum,
        keep_tags=keep_tags,
    )


def create_dataset(subset, **kwargs):
    dataset = BaseDataset(
        resolution=(512, 512),
        network_multiplier=1.0,
        train_inpainting=False,
        debug_dataset=False,
        **kwargs,
    )
    dataset.batch_size = 1
    dataset.subsets = [subset]
    return dataset


class TestFADImplementation(unittest.TestCase):
    def setUp(self):
        random.seed(42)

    def test_fad_frequencies_with_wildcard_uses_exact_expectation(self):
        subset = create_subset(enable_wildcard=True, enable_fad=True)
        dataset = create_dataset(subset)
        info = ImageInfo(
            image_key="img_wildcard",
            num_repeats=1,
            caption="trigger, cat\ntrigger, {dog|bird}",
            is_reg=False,
            absolute_path="/mock/img/dir/img_wildcard.jpg",
        )
        info.image_size = (512, 512)
        dataset.register_image(info, subset)

        dataset.make_buckets()

        freq = subset.fad_tag_frequencies
        self.assertAlmostEqual(freq.get("trigger", 0.0), 1.0)
        self.assertAlmostEqual(freq.get("cat", 0.0), 0.5)
        self.assertAlmostEqual(freq.get("dog", 0.0), 0.25)
        self.assertAlmostEqual(freq.get("bird", 0.0), 0.25)

    def test_fad_curriculum_scales_from_zero_to_one(self):
        subset = create_subset(enable_fad=True, fad_curriculum=True)
        dataset = create_dataset(
            subset,
            fad_curriculum_start=0.0,
            fad_curriculum_end=1.0,
            fad_curriculum_beta=3.0,
            fad_step_start=0.0,
            fad_step_end=1.0,
            fad_p_min=0.35,
            fad_p_max=1.0,
            fad_alpha=10.0,
            fad_c=0.5,
        )
        info = ImageInfo(
            image_key="img",
            num_repeats=1,
            caption="trigger, cat",
            is_reg=False,
            absolute_path="/mock/img/dir/img.jpg",
        )
        info.image_size = (512, 512)
        dataset.register_image(info, subset)
        dataset.make_buckets()
        dataset.set_max_train_steps(1000)

        dataset.set_current_step(0)
        survived = sum("cat" in dataset.process_caption(subset, "trigger, cat") for _ in range(200))
        self.assertEqual(survived, 200)

        dataset.set_current_step(1000)
        survived = sum("cat" in dataset.process_caption(subset, "trigger, cat") for _ in range(500))
        self.assertLess(survived / 500, 0.03)

    def test_dataset_classes_accept_dataset_level_fad_params(self):
        common_params = dict(
            subsets=[],
            batch_size=1,
            resolution=(512, 512),
            network_multiplier=1.0,
            enable_bucket=False,
            min_bucket_reso=256,
            max_bucket_reso=1024,
            bucket_reso_steps=64,
            bucket_no_upscale=False,
            train_inpainting=False,
            debug_dataset=False,
            validation_split=0.0,
            validation_seed=None,
            resize_interpolation=None,
            skip_image_resolution=None,
            fad_p_min=0.2,
            fad_p_max=0.9,
            fad_alpha=8.0,
            fad_c=0.4,
            fad_curriculum_start=0.2,
            fad_curriculum_end=0.7,
            fad_curriculum_beta=2.0,
            fad_step_start=0.0,
            fad_step_end=1.0,
        )

        dreambooth_dataset = DreamBoothDataset(is_training_dataset=True, prior_loss_weight=1.0, **common_params)
        finetuning_dataset = FineTuningDataset(**common_params)
        controlnet_dataset = ControlNetDataset(**common_params)

        self.assertEqual(dreambooth_dataset.fad_p_min, 0.2)

    def test_keep_tags_default_protects_fad_matched_flex_tokens(self):
        subset = create_subset(enable_fad=True)
        dataset = create_dataset(
            subset,
            fad_p_min=1.0,
            fad_p_max=1.0,
            fad_alpha=10.0,
            fad_c=0.5,
        )
        info = ImageInfo(
            image_key="img",
            num_repeats=1,
            caption="trigger, 1girl, cat",
            is_reg=False,
            absolute_path="/mock/img/dir/img.jpg",
        )
        info.image_size = (512, 512)
        dataset.register_image(info, subset)
        dataset.make_buckets()
        dataset.set_max_train_steps(1000)
        dataset.set_current_step(1000)

        caption = dataset.process_caption(subset, "trigger, 1girl, cat")

        self.assertIn("1girl", caption)
        self.assertNotIn("cat", caption)

    def test_keep_tags_custom_regex_protects_tag_dropout_with_flexible_matching(self):
        subset = create_subset(
            keep_tags="fake_screenshot, window\\(computing\\), .* logo",
        )
        subset.caption_tag_dropout_rate = 1.0
        dataset = create_dataset(subset)

        caption = dataset.process_caption(
            subset,
            "trigger, Fake Screenshot, window(computing), artist_logo, cat",
        )

        self.assertIn("Fake Screenshot", caption)
        self.assertIn("window(computing)", caption)
        self.assertIn("artist_logo", caption)
        self.assertNotIn("cat", caption)

    def test_default_keep_tags_patterns_match_flexible_tokens(self):
        subset = create_subset()
        cases = {
            "1girl": True,
            "1GIRL": True,
            "2girls": True,
            "solo": False,
            "cropped_head": True,
            "disembodied_arm": True,
            "artist logo": True,
            "artist_logo": True,
            "window_(computing)": True,
            r"window_\(computing\)": True,
            "fake phone screenshot": True,
            "fake_phone_screenshot": True,
            "speech bubble": True,
            "speech_bubble": True,
            "bad_censor_bar": True,
            "motion blur": True,
            "motion_blur": True,
            "character_profile": True,
            "front_cover": True,
            "multiple_views": True,
            "oil_(medium)": True,
            r"oil_\(medium\)": True,
            "sketch": True,
            "cat": False,
        }

        for token, expected in cases.items():
            with self.subTest(token=token):
                self.assertEqual(subset.is_keep_tag(token), expected)

    def test_tsfad_interpolation(self):
        # Verify interpolation formula (1 - t) * sFAD + t * 1.0
        subset = BaseSubset(
            image_dir="/mock/img/dir",
            alpha_mask=False,
            num_repeats=1,
            shuffle_caption=False,
            caption_separator=",",
            keep_tokens=0,
            keep_tokens_separator=None,
            secondary_separator=None,
            enable_wildcard=False,
            color_aug=False,
            flip_aug=False,
            face_crop_aug_range=None,
            random_crop=False,
            caption_dropout_rate=0.0,
            caption_dropout_every_n_epochs=0,
            caption_tag_dropout_rate=0.0,
            caption_prefix=None,
            caption_suffix=None,
            token_warmup_min=0,
            token_warmup_step=0,
            enable_fad=True,
            fad_curriculum=True,
            fad_timestep=True,
            keep_tags=None,
        )
        dataset = create_dataset(
            subset,
            fad_curriculum_start=0.0,
            fad_curriculum_end=1.0,
            fad_curriculum_beta=3.0,
            fad_step_start=0.0,
            fad_step_end=1.0,
            fad_p_min=1.0,
            fad_p_max=1.0,
            fad_alpha=10.0,
            fad_c=0.5,
        )
        info = ImageInfo(
            image_key="img",
            num_repeats=1,
            caption="trigger, cat",
            is_reg=False,
            absolute_path="/mock/img/dir/img.jpg",
        )
        info.image_size = (512, 512)
        dataset.register_image(info, subset)
        dataset.make_buckets()
        dataset.set_max_train_steps(1000)
        
        # At step 0, sFAD p_step is 0.0 (no dropout)
        dataset.set_current_step(0)
        
        # With t_val = 0.0, TSFAD intensity is 0.0 -> no dropout
        caption = dataset.process_caption(subset, "trigger, cat", t_val=0.0)
        self.assertIn("cat", caption)
        
        # With t_val = 1.0, TSFAD intensity is 1.0 -> full dropout
        caption = dataset.process_caption(subset, "trigger, cat", t_val=1.0)
        self.assertNotIn("cat", caption)


if __name__ == "__main__":
    unittest.main()
