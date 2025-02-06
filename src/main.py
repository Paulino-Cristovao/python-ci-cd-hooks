import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Determine the device to use (MPS, CUDA, or CPU)
device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"Using device: {device}")

# Model and tokenizer initialization
model_name = "deepseek-ai/deepseek-coder-1.3b-base"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name, trust_remote_code=True, torch_dtype=torch.bfloat16
).to(device)


def chat_with_deepseek(input_text: str, max_length: int = 50) -> tuple[str, int]:
    """
    Generates text using the DeepSeek model.

    Args:
        input_text (str): User input query.
        max_length (int): Maximum length of generated text.

    Returns:
        tuple[str, int]: Generated text and word count.
    """
    # Encode the input text
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    # Generate text without gradient calculation
    with torch.no_grad():
        output = model.generate(
            input_ids, max_length=max_length, num_return_sequences=1
        )

    # Decode the generated text
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    word_count = len(output_text.split())

    return output_text, word_count
