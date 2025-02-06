from src.main import chat_with_deepseek


def test_chat_with_deepseek() -> None:
    """
    Test the chat_with_deepseek function to ensure it runs without errors and returns
    the expected types and values.

    This test checks:
    - The function returns a string as the generated text.
    - The function returns an integer as the word count.
    - The generated text is not empty.
    - The word count is greater than zero.
    """
    # Input text for the test
    input_text: str = "What is the capital of Mozambique?"

    # Call the function with the input text
    generated_text: str
    word_count: int
    generated_text, word_count = chat_with_deepseek(input_text)

    # Assertions to validate the function's output
    assert isinstance(generated_text, str), "Output should be a string"
    assert isinstance(word_count, int), "Word count should be an integer"
    assert len(generated_text) > 0, "Should return some text"
    assert word_count > 0, "Word count should be greater than zero"
