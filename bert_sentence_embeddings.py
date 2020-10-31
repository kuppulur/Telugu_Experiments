"""Generate sentence embeddings."""
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelWithLMHead


def load_model():
    """Load model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained("kuppuluri/telugu_bertu",
                                              clean_text=False,
                                              handle_chinese_chars=False,
                                              strip_accents=False,
                                              wordpieces_prefix='##')
    model = AutoModelWithLMHead.from_pretrained("kuppuluri/telugu_bertu",
                                                output_hidden_states=True)
    return model, tokenizer


def get_embedding(sentence, model, tokenizer):
    """Create sentence embedding."""
    input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)
    outputs = model(input_ids)
    hidden_states = outputs[1]

    # get last two layers or change accordingly
    last_two_layers = [hidden_states[i] for i in (-1, -2)]
    concatenated_hidden_states = torch.cat(tuple(last_two_layers), dim=-1)

    sentence_embedding = torch.mean(concatenated_hidden_states,
                                    dim=1).squeeze()

    sentence_embedding = sentence_embedding.data.numpy()
    normalized_sentence_embedding = sentence_embedding / np.linalg.norm(
        sentence_embedding)
    return normalized_sentence_embedding


if __name__ == '__main__':
    sentence = "ర్యాష్‌ డ్రైవింగ్‌ చేసిన సినీ నిర్మాతపై కేసు"
    model, tokenizer = load_model()
    sentence_embedding = get_embedding(sentence, model, tokenizer)
    print(sentence_embedding.shape)  # 768*2 = 1536
