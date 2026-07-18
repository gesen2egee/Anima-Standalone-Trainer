from library.differential_output_preservation import (
    build_preservation_prompt,
    find_caption_class_tokens,
    split_caption_tokens,
)


def test_split_caption_tokens_trims_and_discards_empty_tokens():
    assert split_caption_tokens(" Alice , blue eyes, , outdoors ") == ["Alice", "blue eyes", "outdoors"]


def test_finds_supported_person_count_tags_only_as_complete_tokens():
    caption = "Alice, 1GIRL, 2boys, 6others, 7girls, 1girls, 2girls portrait"
    assert find_caption_class_tokens(caption) == ["1girl", "2boys", "6others"]


def test_empty_dataset_class_uses_caption_class_tags_without_flex_tokens():
    caption = "Vegapunk_York, 1girl, blue eyes, smile"
    assert build_preservation_prompt(caption) == "1girl"


def test_dataset_class_overrides_detected_caption_tags():
    assert build_preservation_prompt("Alice, 1girl, outdoors", class_tokens="woman") == "woman"


def test_missing_dataset_and_caption_class_skips_preservation():
    assert build_preservation_prompt("Alice, outdoors, smiling") is None


def test_custom_separator_is_supported():
    assert build_preservation_prompt("Alice|2girls|blue eyes", "|") == "2girls"
