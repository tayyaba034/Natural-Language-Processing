import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import string
import heapq
from transformers import pipeline

def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')

download_nltk_data()

def summarize_extractive(text, num_sentences=3):
    if not text:
        return ""
    sentences = sent_tokenize(text)

    if len(sentences) <= num_sentences:
        return text

    stop_words = set(stopwords.words("english"))
    word_frequencies = {}

    for word in word_tokenize(text):
        word = word.lower()
        if word not in stop_words and word not in string.punctuation:
            if word not in word_frequencies:
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1

    if not word_frequencies:
        return text

    max_frequency = max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word] /= max_frequency

    sentence_scores = {}
    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word in word_frequencies:
                if sentence not in sentence_scores:
                    sentence_scores[sentence] = word_frequencies[word]
                else:
                    sentence_scores[sentence] += word_frequencies[word]

    summary_sentences = heapq.nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
    summary_sentences.sort(key=lambda s: sentences.index(s))
    
    return ' '.join(summary_sentences)

#Global summarizer pipeline to avoid reloading
_abstractive_summarizer = None

def get_abstractive_summarizer():
    global _abstractive_summarizer
    if _abstractive_summarizer is None:
        _abstractive_summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    return _abstractive_summarizer

def summarize_abstractive(text, max_length=130, min_length=30):
    if not text or len(text.strip()) < 50:
        return text
    
    summarizer = get_abstractive_summarizer()
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']

def summarize_all(text):
    return {
        'extractive': summarize_extractive(text),
        'abstractive': summarize_abstractive(text)
    }

if __name__ == "__main__":
    test_text = """
    Artificial Intelligence (AI) is transforming industries across the world by enabling machines to perform tasks that traditionally required human intelligence.
    These tasks include problem-solving, decision-making, language understanding, and visual perception. 
    AI systems rely on large amounts of data, advanced algorithms, and powerful computing resources to learn patterns and make predictions. 
    While AI offers significant benefits such as increased efficiency, accuracy, and automation, it also raises concerns related to data privacy, 
    job displacement, and ethical decision-making. As AI continues to evolve, it is important for governments, organizations, and individuals to ensure its responsible
    and fair use for the benefit of society.
    """
    print("Extractive Summary: ")
    print(summarize_extractive(test_text))
    print("\n Abstractive Summary: ")
    print(summarize_abstractive(test_text))
