import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest import mock

from PIL import Image


class MultilabelTimmUtilsTest(unittest.TestCase):
    def test_tags_to_text_matches_imgutils_spacing_without_escape(self):
        from library.multilabel_timm import tags_to_text

        tags = {
            "panty_pull": 0.6826801300048828,
            "panties": 0.958938717842102,
            "drinking_glass": 0.9340789318084717,
            "hu_tao_(genshin_impact)": 0.8973554372787476,
            "1girl": 0.9988248348236084,
        }

        self.assertEqual(
            tags_to_text(tags, use_spaces=True, use_escape=False),
            "1girl, panties, drinking glass, hu tao (genshin impact), panty pull",
        )

    def test_split_hf_repo_id_supports_model_subfolder(self):
        from library.multilabel_timm import split_hf_repo_id

        self.assertEqual(
            split_hf_repo_id("Makki2104/animetimm/eva02_large_patch14_448.dbv4-full"),
            ("Makki2104/animetimm", "eva02_large_patch14_448.dbv4-full"),
        )
        self.assertEqual(
            split_hf_repo_id("animetimm/mobilenetv3_large_150d.dbv4-full"),
            ("animetimm/mobilenetv3_large_150d.dbv4-full", None),
        )

    def test_compose_caption_uses_char_rating_general_order(self):
        from library.multilabel_timm import compose_caption_text

        caption = compose_caption_text(
            rating={"sensitive": 0.9, "general": 0.1},
            general={"long_hair": 0.8, "1girl": 0.95},
            character={"hu_tao_(genshin_impact)": 0.99},
            include_char=True,
            include_rating=True,
            include_general=True,
        )

        self.assertEqual(
            caption,
            "hu tao (genshin impact), sensitive, 1girl, long hair",
        )

    def test_create_pillow_transforms_supports_pad_to_size(self):
        from library.multilabel_timm import create_pillow_transforms

        transform = create_pillow_transforms({
            "type": "pad_to_size",
            "size": [8, 8],
            "background_color": "white",
            "interpolation": "bilinear",
        })

        image = Image.new("RGB", (8, 4), (0, 0, 0))
        padded = transform(image)

        self.assertEqual(padded.size, (8, 8))
        self.assertEqual(padded.getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(padded.getpixel((4, 4)), (0, 0, 0))

    def test_main_unloads_models_after_caption_writer_finishes(self):
        from library import multilabel_timm

        with mock.patch.object(
            multilabel_timm,
            "write_captions_for_directory",
            return_value={"total": 0, "written": 0, "failed": 0},
        ), mock.patch.object(multilabel_timm, "unload_multilabel_timm_models") as unload:
            with redirect_stdout(StringIO()):
                exit_code = multilabel_timm.main(["--image-dir", "."])

        self.assertEqual(exit_code, 0)
        unload.assert_called_once()

    def test_unload_clears_model_cache_and_instances(self):
        from library import multilabel_timm

        model = multilabel_timm._open_model_for_repo("owner/repo")
        self.assertIs(multilabel_timm._open_model_for_repo("owner/repo"), model)
        self.assertGreater(multilabel_timm._open_model_for_repo.cache_info().currsize, 0)
        self.assertTrue(multilabel_timm._OPEN_MODEL_INSTANCES)

        multilabel_timm.unload_multilabel_timm_models()

        self.assertEqual(multilabel_timm._open_model_for_repo.cache_info().currsize, 0)
        self.assertFalse(multilabel_timm._OPEN_MODEL_INSTANCES)


if __name__ == "__main__":
    unittest.main()
