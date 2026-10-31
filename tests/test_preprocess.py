from app.preprocess import preprocess, tokenize


def test_tokenize_lowercases_and_splits_punctuation():
    assert tokenize("Hello, World! It's 2024.") == ["hello", "world", "it", "s", "2024"]


def test_stop_words_and_single_characters_are_removed():
    assert preprocess("The cat and a dog") == preprocess("cat dog")
    assert preprocess("the and of to") == []


def test_stemming_conflates_word_forms():
    assert preprocess("connect")[0] == preprocess("connected")[0] == preprocess("connection")[0]


def test_query_and_document_share_one_pipeline():
    assert preprocess("Markets FALL") == preprocess("markets fall")
