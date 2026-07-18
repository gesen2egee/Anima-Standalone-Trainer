from library.differential_output_preservation import (
    build_preservation_prompt,
    contains_trigger_word,
    normalize_trigger_match,
    split_caption_tokens,
)


def test_split_caption_tokens_trims_and_discards_empty_tokens():
    assert split_caption_tokens(" Alice , blue eyes, , outdoors ") == ["Alice", "blue eyes", "outdoors"]


def test_trigger_matching_ignores_case_space_underscore_and_parenthesis_escaping():
    trigger = r"Vegapunk_York \(One Piece\)"
    assert normalize_trigger_match(trigger) == normalize_trigger_match("vegapunk york (one_piece)")
    assert contains_trigger_word("portrait of VEGAPUNK YORK (ONE PIECE)", [trigger])


def test_classless_prompt_keeps_only_flex_tokens_without_triggers():
    caption = r"Vegapunk_York, blue eyes, vegapunk york outfit, smile, \(Vegapunk York\)"
    assert build_preservation_prompt(caption, keep_tokens=1) == "blue eyes, smile"


def test_class_prompt_replaces_trigger_section_and_keeps_safe_flex_tokens():
    caption = "Alice, Bob, outdoors, alice costume, smiling"
    assert build_preservation_prompt(caption, keep_tokens=2, preservation_class="woman, man") == (
        "woman, man, outdoors, smiling"
    )


def test_custom_separator_is_supported():
    assert build_preservation_prompt("Alice|blue eyes|ALICE_dress", 1, "|", "woman") == "woman| blue eyes"
