"""Demo script for showcasing Telugu Language model."""
import streamlit as st
from transformers import AutoModelWithLMHead, AutoTokenizer, pipeline


@st.cache(allow_output_mutation=True)
def load_pipeline():
    """Load custom built Telugu Language model pipeline."""
    tokenizer = AutoTokenizer.from_pretrained("kuppuluri/telugu_bertu",
                                              clean_text=False,
                                              handle_chinese_chars=False,
                                              strip_accents=False,
                                              wordpieces_prefix='##')

    model = AutoModelWithLMHead.from_pretrained("kuppuluri/telugu_bertu")
    fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)
    return fill_mask


def main():
    """BERT Telugu Language modeling demo."""
    st.sidebar.title("""

        My Custom Model.

        Example sentences:

        1. మక్దూంపల్లి పేరుతో చాలా [MASK] ఉన్నాయి.

        2. నోరూరించే వంకాయ-జీడిపప్పు [MASK] చేయు విధానం

        3. [MASK] దెబ్బకు హెచ్‌సిఎల్ ఉద్యోగులకు వర్క్ ప్రమ్ హొమ్ అవకాశం

        4. నిజం చెప్పొద్దూ, [MASK] మొహాలు ఒక్కసారి మతాబాల్లాగ వెలిగిపోయాయి

        5. ఆశ్రమంలో వాతవరణం అంతా [MASK] వుంది.

        6. [MASK] లోకి రాగానే  సిగరెట్ కాల్చాలని పించింది.

        7. ఆస్ట్రేలియా [MASK] టెన్నిస్ టోర్నమెంటులో సంచలనం సానియా మీర్జా మూడో రౌండులోకి ప్రవేశించింది.

        8. ప్రస్తుత [MASK] సంవత్సరానికి గాను మొదటి త్రైమాసికంలో ఏప్రిల్ - జూన్  లో పరోక్ష పన్నులు 13.8 శాతానికి పెరిగి రూ.1.11 లక్షల కోట్లకు చేరంది.

        9. భారీ [MASK] ఆర్జిస్తూ, నవరత్న హోదా కలిగిన కేంద్ర ప్రభుత్వ రంగ సంస్థ స్టీల్  అథారిటీ ఆఫ్ ఇండియా

        10. శ్రీగిరిపల్లి, [MASK] రాష్ట్రం, సిద్ధిపేట జిల్లా, గజ్వేల్ మండలంలోని గ్రామం.

        11. భీమావరం పట్టణంలో ఒక చిన్న [MASK] స్టేషన్

        """)

    text = st.text_area("Text (టెక్స్ట్)",
                        "మక్దూంపల్లి పేరుతో చాలా [MASK] ఉన్నాయి.")
    fill_mask = load_pipeline()

    if st.button("Fill masks"):
        results = fill_mask(text)
        for result in results:
            score = "{0:.3%}".format(result["score"])
            st.markdown("<p>" + "<span style='color:#808080'>" + score +
                        "</span>" + "  " + result["sequence"] + "</p>",
                        unsafe_allow_html=True)


if __name__ == '__main__':
    main()
