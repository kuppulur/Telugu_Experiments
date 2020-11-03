# Telugu_Experiments

In this repository, I demo some of the applications of language modeling for Telugu using a language model I trained on various Telugu Corpora.

https://karthikuppuluri.medium.com/language-modeling-for-%E0%B0%A4%E0%B1%86%E0%B0%B2%E0%B1%81%E0%B0%97%E0%B1%81-telugu-b590a029a565


# Installation (Code is tested on Python 3.7.3)

```pip install -r requirements.txt```

I will not actively maintain this repository. Please expect delays in my response if there are any issues posted.

# 1. Language Model Server

``` streamlit run language_model_demo.py```

![Image](https://github.com/kuppulur/Telugu_Experiments/blob/main/images/language_model.png)


# 2. Fasttext Demo

My Model Download Link: https://drive.google.com/drive/folders/1kIDJsSyesbLn42J6Fqa-kKuhhsR-efc-?usp=sharing

Pre-trained Telugu Model from Fasttext is available here: https://fasttext.cc/docs/en/crawl-vectors.html 

``` streamlit run fasttext_telugu_demo.py```

![Image](https://github.com/kuppulur/Telugu_Experiments/blob/main/images/neighbors_from_fasttext.png)


# 3. POS and NER demo

``` streamlit run pos_and_ner_demo.py```

![Image](https://github.com/kuppulur/Telugu_Experiments/blob/main/images/pos_and_ner.png)


# 4. Comprehension Demo

``` streamlit run comprehension_demo_telugu.py```

![Image](https://github.com/kuppulur/Telugu_Experiments/blob/main/images/comprehension.png)


# 5. Summarizer Demo

``` streamlit run summarizer_demo.py```

![Image](https://github.com/kuppulur/Telugu_Experiments/blob/main/images/summarizer.png)


# 6. BERT Telugu Sentence Embeddings
``` python bert_sentence_embeddings.py```


# 7. Semantic Search
``` python semantic_search_example.py```

![Image](https://github.com/kuppulur/Telugu_Experiments/blob/main/images/semantic_result.png)


# 8. Clustering
``` python clustering_example.py```

![Image](https://github.com/kuppulur/Telugu_Experiments/blob/main/images/cluster_result.png)
