"""Generate clusters from corpus."""
# reference: https://github.com/UKPLab/sentence-transformers/blob/master/examples/applications/semantic_search.py
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


def get_similarities(query_vector, corpus_vectors):
    """Cosine similarity."""
    rep_query = np.tile(query_vector, (len(corpus_vectors), 1))
    cosine = rep_query * np.array(corpus_vectors)
    cosine = np.sum(cosine, axis=1)
    return cosine.tolist()


if __name__ == '__main__':

    corpus = [
        "రుచికరమైన టొమాటో గొజ్జు రిసిపి",
        "సీఎం కేసీఆర్‌పై ఎల్లలు దాటిన అభిమానం",
        "ఏపీలో స్థానిక ఎన్నికల కసరత్తు.. రేపు ఈసీ అఖిలపక్ష భేటీ",
        "నిరూపిస్తే నిమిషంలో రాజీనామా చేస్తా: కేసీఆర్‌",
        "బ్యాటింగ్‌లో కోహ్లీసేన అట్టర్ ఫ్లాఫ్.. హైదరాబాద్ ముందు ఈజీ టార్గెట్!",
        "ఏపీలో స్థానిక సంస్థ‌ల ఎన్నిక‌ల‌కు రాష్ట్ర ఎన్నిక‌ల సంఘం మొగ్గుచూపుతున్న క్ర‌మంలో",
        "అందుకే అతని బౌలింగ్ ఒకసారి ఆడాలనుకుంటున్నా: సచిన్ టెండూల్కర్",
        "కొత్త స్మార్ట్‌ఫోన్‌లు వచ్చేసాయి!!! ఫీచర్స్ బ్రహ్మాండం",
        "కోహ్లీసేనదే బ్యాటింగ్.. ఇరు జట్లలో మార్పులు!",
        "మరో రెండు కొత్త ఫోన్లు ? ధరలు మరియు ఫీచర్లు చూడండి.",
        "క్యాప్సికం మసాలా గ్రేవీ రిసిపి",
        "25 వేల లోపు ఈ స్మార్ట్‌ఫోన్‌ల మీద ఊహించని డిస్కౌంట్ ఆఫర్స్!",
        "సీఎం కేసీఆర్‌పై అభిమానానికి ఎల్లలు లేవు. తెలంగాణ సాధించిన",
        "రుచికరమైన కీమా దాళ్ రిసిపి: పరాఠా, చపాతీ, నాన్ మరియు రోటీలకు అద్భుతమైన కాంబినేషన్"
    ]

    query_sentences = [
        "నిరూపిస్తే దుబ్బాక చౌరస్తాలో ఉరివేసుకుంటా..సీఎం కేసీఆర్ కి బండి సంజయ్ సవాల్",
        "టమోటా మసాలా గ్రేవీ రిసిపి", "స్థానిక ఎన్నికల అంశం",
        "సచిన్ టెండూల్కర్ బ్యాక్ ఫుట్ డిఫెన్సె కోచ్ ఎవరో తెలుసా"
    ]

    # load model
    model, tokenizer = load_model()

    # get embeddings
    corpus_vectors = [
        get_embedding(sentence, model, tokenizer) for sentence in corpus
    ]

    for query in query_sentences:
        query_vector = get_embedding(query, model, tokenizer)
        similarities_to_corpus = get_similarities(query_vector, corpus_vectors)
        sorted_similarities_index = np.argsort(similarities_to_corpus)[::-1]
        sorted_corpus = [corpus[index] for index in sorted_similarities_index]
        sorted_similarities = [
            similarities_to_corpus[index]
            for index in sorted_similarities_index
        ]
        topk = 3
        print("\nQuery of interest:", query)
        print("\nTop 3 most similar sentences in corpus:")
        for score, sentence in zip(sorted_similarities[0:topk],
                                   sorted_corpus[0:topk]):
            print(sentence, "(Score: %.4f)" % (score))
