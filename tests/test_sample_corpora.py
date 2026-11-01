from scripts.sample_corpora import _s1_filter_reason


def test_s1_filter_rejects_extreme_output_request():
    assert _s1_filter_reason("Write a 50,000-word romance novel.") == "s1_extreme_output"


def test_s1_filter_rejects_software_build_task():
    prompt = "Design and develop an online booking application for classes."

    assert _s1_filter_reason(prompt) == "s1_software_task"


def test_s1_filter_rejects_site_game_and_language_specific_code_tasks():
    prompts = [
        "Build a responsive social networking site with messaging.",
        "Develop a 2D mobile game using Swift and SpriteKit.",
        "Create a Go program that converts decimal to hexadecimal.",
    ]

    assert all(_s1_filter_reason(prompt) == "s1_software_task" for prompt in prompts)


def test_s1_filter_keeps_bounded_general_prompt():
    assert _s1_filter_reason("Explain why leaves change color in autumn.") is None


def test_s1_filter_keeps_non_software_program_prompt():
    prompt = "Develop an onboarding program for new employees."

    assert _s1_filter_reason(prompt) is None


def test_s1_filter_ignores_comma_before_the_word_words():
    assert _s1_filter_reason("Describe colors, words, and punctuation in a poem.") is None