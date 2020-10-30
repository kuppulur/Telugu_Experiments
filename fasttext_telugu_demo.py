"""Demo script for showcasing Telugu Fasttext model."""
import fasttext
import pandas as pd
import streamlit as st


@st.cache(allow_output_mutation=True)
def load_fasttext_model():
    """Load telugu fasttext model."""
    my_custom_fasttext_model = fasttext.load_model("telugu_original.bin")
    pre_trained_fasttext_model = fasttext.load_model("cc.te.300.bin")
    return my_custom_fasttext_model, pre_trained_fasttext_model


def main():
    """Fasttext Telugu model nearest neighbors demo."""
    st.sidebar.title("""

        Telugu Fasttext Model Comparison Demo.

        """)

    text = st.text_area("Please enter your word (టెక్స్ట్)", "స్టాక్")
    my_custom_fasttext_model, pre_trained_fasttext_model = load_fasttext_model(
    )

    if st.button("Get nearest neighbors"):
        st.title("Results from my custom fasttext model")
        results = my_custom_fasttext_model.get_nearest_neighbors(text, k=25)
        st.table(pd.DataFrame(list(results)))
        st.title("Results from pre-trained fasttext model")
        results = pre_trained_fasttext_model.get_nearest_neighbors(text, k=25)
        st.table(pd.DataFrame(list(results)))


if __name__ == '__main__':
    main()
