"""Summarization demo for Telugu."""
import streamlit as st
from summarizer import Summarizer
from transformers import BertModel, AutoTokenizer, AutoConfig

st.header("Extractive Summarization Demo")


@st.cache(allow_output_mutation=True)
def load_model():
    """Load custom built Telugu Language model pipeline for summarization."""
    custom_config = AutoConfig.from_pretrained("kuppuluri/telugu_bertu",
                                               output_hidden_states=True)
    tokenizer = AutoTokenizer.from_pretrained("kuppuluri/telugu_bertu",
                                              clean_text=False,
                                              handle_chinese_chars=False,
                                              strip_accents=False,
                                              wordpieces_prefix='##')

    model = BertModel.from_pretrained("kuppuluri/telugu_bertu",
                                      config=custom_config)
    summarization_model = Summarizer(custom_model=model,
                                     custom_tokenizer=tokenizer)
    return summarization_model


def get_summary(context, model):
    """Get summary from the context."""
    return model(context)


def main():
    """Telugu Summarization demo."""
    context = st.text_area(
        "Context (ప్రకరణము)",
        "జగన్ సర్కార్‌కు కేంద్రం గుడ్‌న్యూస్ చెప్పింది. హైదరాబాద్‌ నుంచి తిరుపతి, చెన్నైలకు తక్కువ సమయంలోనే వెళ్లేందుకు కడప–రేణిగుంట మధ్య నాలుగు వరుసల హైవేకు గ్రీన్ సిగ్నల్ వచ్చింది. రాయలసీమ జిల్లాలకు ముఖ్యమైన ఈ రోడ్డు రెండు వరుసల నుంచి నాలుగు లేన్లుగా మార్చేందుకు నేషనల్‌ హైవేస్‌ అథారిటీ ఆఫ్‌ ఇండియా (ఎన్‌హెచ్‌ఏఐ) త్వరలో టెండర్లకు కూడా సిద్ధమవుతున్నారు. ఈ హైవేను కేంద్రం ఇటీవలే గ్రీన్‌ఫీల్డ్‌ ఎక్స్‌ప్రెస్‌ వేగా గుర్తించింది..ఒక్క కడప జిల్లాలోనే సుమారు 100 కి.మీ. మేర రహదారి నిర్మించనున్నారు. దీనిని రెండు ప్యాకేజీలుగా విభజించి 1,068 ఎకరాలు సేకరించనున్నారు. వైఎస్సార్‌ జిల్లా బద్వేలు నుంచి నెల్లూరు జిల్లా కృష్ణపట్నం పోర్టు వరకు 4 లేన్ల రహదారి నిర్మాణానికి డీపీఆర్‌ సిద్ధమైంది. మొత్తం 138 కి.మీ. మేర రోడ్డు నిర్మాణాన్ని ఎన్‌హెచ్‌ఏఐ చేపట్టనుంది. ఈ మార్గంలో 3 వంతెనలు, 2 రైల్వే ఓవర్‌ బ్రిడ్జిలు నిర్మించనున్నారు. రెండో ప్యాకేజీ కింద కడప జిల్లా సిద్ధవటం మండలం నుంచి రైల్వేకోడూరు మండలం వరకు నిర్మించేందుకు కసరత్తు చేస్తున్నారు. ఈ నాలుగు లేన్ల హైవేకు భూ సేకరణ పనులు ముమ్మరం చేశారు. గతేడాది అక్టోబర్‌లో ఈ హైవేకు ఎన్‌హెచ్‌–716 కేటాయించారు. కడప దగ్గర వైఎస్సార్‌ టోల్‌ప్లాజా నుంచి రేణిగుంట వరకు 4 లేన్ల నిర్మాణం జరగనుంది. రూ.3 వేల కోట్లతో 133 కి.మీ. మేర నిర్మించనున్న ఈ హైవే నిర్మాణానికి కేంద్రం అంగీకరించడంతో జిల్లా అధికార యంత్రాంగం అన్ని ఏర్పాట్లు వేగవంతంగా చేస్తోంది. నాలుగు వరుసల ఈ హైవే టెండర్లను త్వరలోనే పూర్తిచేస్తామని అధికారులు అంటున్నారు. ఈ ప్రాజెక్టును నాలుగేళ్లలో నిర్మిస్తామని.. ఇప్పటికే భూసేకరణ పనులు ప్రారంభించామంటున్నారు.",
        height=100)

    summarization_model = load_model()

    if st.button('summarize'):
        result = summarization_model(context)
        st.markdown("<b>In-Short</b>", unsafe_allow_html=True)
        st.markdown("<p>" + "<mark>" + result + "</mark>" + "</p>",
                    unsafe_allow_html=True)


if __name__ == '__main__':
    main()
