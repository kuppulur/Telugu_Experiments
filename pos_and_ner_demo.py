"""."""
import spacy
from spacy import displacy
import numpy as np
import streamlit as st
from scipy.special import softmax
from simpletransformers.ner import NERModel
from spacy.gold import iob_to_biluo, offsets_from_biluo_tags

HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""


def get_token_for_char(doc, char_idx):
    """Get the token id from character index."""
    for i, token in enumerate(doc):
        if char_idx > token.idx:
            continue
        if char_idx == token.idx:
            return i
        if char_idx < token.idx:
            return i - 1
    return len(doc) - 1


@st.cache(allow_output_mutation=True)
def load_models():
    """Load POS and NER trained telugu models."""
    pos_model = NERModel('bert',
                         'kuppuluri/telugu_bertu_pos',
                         args={"use_multiprocessing": False},
                         labels=[
                             'QC', 'JJ', 'NN', 'QF', 'RDP', 'O',
                             'NNO', 'PRP', 'RP', 'VM', 'WQ',
                             'PSP', 'UT', 'CC', 'INTF', 'SYMP',
                             'NNP', 'INJ', 'SYM', 'CL', 'QO',
                             'DEM', 'RB', 'NST', ],
                         use_cuda=False)

    ner_model = NERModel('bert',
                         'kuppuluri/telugu_bertu_ner',
                         labels=[
                             'B-PERSON', 'I-ORG', 'B-ORG', 'I-LOC', 'B-MISC',
                             'I-MISC', 'I-PERSON', 'B-LOC', 'O'
                         ],
                         use_cuda=False,
                         args={"use_multiprocessing": False})

    spacy_telugu_model = spacy.blank("te")

    return pos_model, ner_model, spacy_telugu_model


def format_predictions_to_display(doc,
                                  predictions,
                                  probability_maps,
                                  pos=False):
    """Format predictions into spacy display formar."""
    bert_predictions = []
    iob_tags = []
    tags_formatted = []

    for prediction, probability_map in zip(predictions[0],
                                           probability_maps[0]):
        word = list(prediction.keys())[0]
        probas = probability_map[word]
        normalized_probas = list(softmax(np.mean(probas, axis=0)))
        bert_predictions.append(
            (word, prediction[word], np.max(normalized_probas)))
        if pos:
            iob_tags.append("I-" + prediction[word])
        else:
            iob_tags.append(prediction[word])

    biluo_tags = iob_to_biluo(iob_tags)
    tags = offsets_from_biluo_tags(doc, biluo_tags)

    for tag in tags:
        start_token = get_token_for_char(doc, tag[0])
        word_span = doc.text[tag[0]:tag[1]]
        length_of_span = len(word_span.split())
        if length_of_span == 1:
            probs = [bert_predictions[start_token][2]]
        else:
            probs = [
                item[2] for item in bert_predictions[start_token:start_token +
                                                     length_of_span]
            ]
        tags_formatted.append({
            "start": tag[0],
            "end": tag[1],
            "label": tag[2],
            "score": np.prod(probs)
        })
    return bert_predictions, tags_formatted


def main():
    """BERT Telugu POS and NER model demo."""
    st.sidebar.title("""

        POS and NER model demo.

        Example sentences:

        1. కాంగ్రెస్‌ పార్టీకి గుడ్‌ బై చెప్పి ఇటీవల టీఆర్‌ ఎస్‌ తీర్థం పుచ్చుకున్న డీఎస్‌ కు కేసీఆర్‌ ఈ పదవినిచ్చి గౌరవించారు .

        2. విరాట్ కోహ్లీ కూడా అదే నిర్లక్ష్యాన్ని ప్రదర్శించి కేవలం ఒక పరుగుకే రనౌటై పెవిలియన్ చేరాడు .

        3. లాలూకు తోడు ఇప్పుడు నితీష్‌ కుమార్ కూడా ఈ సభకు హాజరు కాకూడదని నిర్ణయించుకోవటంతో మహాకూటమిలో నెలకొన్న విభేదాలు తార స్థాయికి చేరుకున్నాయని అంటున్నారు .
        """)

    st.sidebar.title("""
        Legend for POS and NER:

        http://universaldependencies.org/docs/en/pos/all.html

        LOC: LOCATION
        PERSON: PERSON
        ORG: ORGANIZATION
        MISC: MISCELLANEOUS

        """)

    text = st.text_area("Text (టెక్స్ట్)",
                        "హైదరాబాద్ లో కిడ్నాప్ కాపాడిన ఏపీ పోలీస్")
    pos_model, ner_model, nlp = load_models()

    if st.button("Get POS and NER"):

        doc = nlp(text)

        pos_predictions, pos_probability_map = pos_model.predict([text])
        ner_predictions, ner_probability_map = ner_model.predict([text])

        bert_pos_predictions, pos_tags_formatted = format_predictions_to_display(
            doc, pos_predictions, pos_probability_map, pos=True)
        bert_ner_predictions, ner_tags_formatted = format_predictions_to_display(
            doc, ner_predictions, ner_probability_map)

        pos_for_display = [{
            "text": doc.text,
            "ents": pos_tags_formatted,
            "title": None
        }]
        ner_for_display = [{
            "text": doc.text,
            "ents": ner_tags_formatted,
            "title": None
        }]

        st.title("Named Entity Results")
        html_ner = displacy.render(ner_for_display, style="ent", manual=True)
        html_ner = html_ner.replace("\n", " ")
        st.write(HTML_WRAPPER.format(html_ner), unsafe_allow_html=True)

        st.title("Part of Speech Results")
        html_pos = displacy.render(pos_for_display, style="ent", manual=True)
        html_pos = html_pos.replace("\n", " ")
        st.write(HTML_WRAPPER.format(html_pos), unsafe_allow_html=True)


if __name__ == '__main__':
    main()
