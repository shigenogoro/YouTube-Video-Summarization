import pandas as pd
from tqdm.auto import tqdm
import os
import re
import spacy

# --- SpaCy NLP setup ---
# Initialize the spaCy model once when the module is imported
try:
    nlp = spacy.load("en_core_web_sm")
    print("SpaCy model 'en_core_web_sm' loaded successfully for preprocessing.")
except OSError:
    print("SpaCy model 'en_core_web_sm' not found. Please ensure it is installed (e.g., run 'python -m spacy download en_core_web_sm').")
    # Set nlp to None to prevent subsequent function calls from crashing on an uninitialized variable
    nlp = None 
        
# --- Core Preprocessing Functions ---

def clean_text(text):
    """Performs basic text cleaning: removing newlines, extra spaces, ellipses, and filler words."""
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r'\b(uh|um|you know|like)\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def split_sentences(text):
    """Splits text into sentences using spaCy, discarding very short sentences."""
    if nlp is None:
        raise RuntimeError("SpaCy NLP model is not loaded. Cannot split sentences.")
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 5]

def chunk_sentences(sentences, max_len=10):
    """Groups a list of sentences into larger chunks."""
    chunks = []
    for i in range(0, len(sentences), max_len):
        chunk = " ".join(sentences[i:i+max_len])
        chunks.append(chunk)
    return chunks

def preprocess_transcript(text):
    """Applies the full preprocessing pipeline (clean, split, chunk) to a single transcript."""
    text = clean_text(text)
    sentences = split_sentences(text)
    chunks = chunk_sentences(sentences)
    return chunks

def preprocess_dataset(dataset_split, output_dir, max_rows_per_file=10000):
    """
    Processes an entire HuggingFace dataset split, creating transcript chunks and saving 
    them to CSV files with their corresponding original summary.
    """
    os.makedirs(output_dir, exist_ok=True)
    preprocessed_data_list = []
    file_index = 0

    for instance in tqdm(dataset_split, desc=f"Preprocessing dataset to {output_dir}"):
        instance_id = instance['id']
        transcript = instance['transcript']
        summary = instance['summary']

        # Call the core preprocessing function
        preprocessed_chunks = preprocess_transcript(transcript)

        for chunk in preprocessed_chunks:
            preprocessed_data_list.append({'id': instance_id, 'transcript': chunk, 'summary': summary})

        # Save to file if batch size is reached
        if len(preprocessed_data_list) >= max_rows_per_file:
            df = pd.DataFrame(preprocessed_data_list)
            output_path = os.path.join(output_dir, f"preprocessed_data_{file_index}.csv")
            df.to_csv(output_path, index=False)
            print(f"Saved {len(df)} rows to {output_path}")
            preprocessed_data_list = []
            file_index += 1

    # Save any remaining data
    if preprocessed_data_list:
        df = pd.DataFrame(preprocessed_data_list)
        output_path = os.path.join(output_dir, f"preprocessed_data_{file_index}.csv")
        df.to_csv(output_path, index=False)
        print(f"Saved {len(df)} rows to {output_path}")