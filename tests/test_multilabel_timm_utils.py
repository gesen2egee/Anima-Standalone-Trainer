import unittest


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


if __name__ == "__main__":
    unittest.main()
