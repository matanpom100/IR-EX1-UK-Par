#!/usr/bin/env python
# coding: utf-8

# 

# In[ ]:





# In[1]:


import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from tqdm import tqdm

def download_debate_files():
    """
    Downloads UK Parliament debate files from the specified start file.
    """
    # כתובות וקבצים לפי הגדרות התרגיל
    base_url = "https://www.theyworkforyou.com/pwdata/scrapedxml/debates/"  # 
    start_file = "debates2023-06-28d.xml"  # 
    output_dir = "data.xml_files"

    # יצירת תיקיית פלט אם היא לא קיימת
    os.makedirs(output_dir, exist_ok=True)

    print(f"Connecting to {base_url} to get the file list...")

    try:
        # 1. קבלת רשימת כל הקבצים מהעמוד
        response = requests.get(base_url)
        response.raise_for_status()  # בדיקה שהבקשה הצליחה
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 2. סינון הרשימה לקבצי XML בלבד
        all_xml_files = []
        for link in soup.find_all('a'):
            href = link.get('href')
            if href and href.endswith('.xml'):
                all_xml_files.append(href)
        
        print(f"Found {len(all_xml_files)} total XML files on the server.")

        # 3. איתור קובץ ההתחלה וסינון הרשימה הסופית
        try:
            start_index = all_xml_files.index(start_file)
            files_to_download = all_xml_files[start_index:]
            print(f"Found start file. Preparing to download {len(files_to_download)} files...")
        except ValueError:
            print(f"Error: Could not find the starting file '{start_file}' in the list.")
            print("Please check the file name and try again.")
            return

        # 4. הורדת הקבצים
        # שימוש ב-tqdm כדי להציג מד התקדמות
        for filename in tqdm(files_to_download, desc="Downloading files", unit="file"):
            file_url = urljoin(base_url, filename)
            local_path = os.path.join(output_dir, filename)

            # בדיקה אם הקובץ כבר קיים כדי למנוע הורדה כפולה
            if os.path.exists(local_path):
                continue
            
            try:
                # הורדת הקובץ
                file_response = requests.get(file_url)
                file_response.raise_for_status()
                
                # שמירת הקובץ
                with open(local_path, 'wb') as f:
                    f.write(file_response.content)
            
            except requests.exceptions.RequestException as e:
                print(f"Failed to download {filename}: {e}")

        print("\nDownload complete!")
        print(f"All files are saved in the '{output_dir}' directory.")

    except requests.exceptions.RequestException as e:
        print(f"Failed to access the website {base_url}: {e}")

if __name__ == "__main__":
    download_debate_files()


# In[3]:


import xml.etree.ElementTree as ET
import os
import re
from collections import defaultdict

# הגדרות נתיבים
XML_DIR = os.path.join('data', 'xml_files')
OUTPUT_DIR = os.path.join('data', 'combined_xml_files')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- הפונקציה is_redirect_file() הוסרה ---

def extract_text_from_xml(xml_file_path):
    """מחזיר את כל הטקסט הגולמי מתוך תגי <p> בלבד."""
    all_text = []
    
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()

        # --- שינוי ---
        # אנחנו כבר לא מסננים. אנחנו פשוט מחפשים תגי <p>.
        # אם בקובץ אין תגי <p> (למשל, קובץ הפניה טהור),
        # הלולאה פשוט תדלג, והפונקציה תחזיר טקסט ריק.
        # אם יש תגי <p> (כמו בקובץ המעורב), הם יחולצו.
        # --- סוף שינוי ---

        for p_tag in root.findall('.//p'):
            if p_tag.text and p_tag.text.strip():
                all_text.append(p_tag.text.strip())

    except ET.ParseError as e:
        print(f"שגיאת ניתוח XML בקובץ {xml_file_path}: {e}")
        return ""
    except Exception as e:
        print(f"שגיאה כללית בקובץ {xml_file_path}: {e}")
        return ""
        
    return ' '.join(all_text)

# --- שאר הלוגיקה זהה לחלוטין ---

combined_texts = defaultdict(list)
file_list = os.listdir(XML_DIR)

print(f"נמצאו {len(file_list)} קבצים. מתחילים בחילוץ ואיחוד...")

for filename in file_list:
    if not filename.endswith('.xml'):
        continue
        
    file_path = os.path.join(XML_DIR, filename)
    
    match = re.search(r'debates(\d{4}-\d{2}-\d{2})[a-zA-Z]*d?\.xml$', filename)
    if not match:
        continue
        
    base_date = match.group(1)
    
    # הפונקציה עכשיו מחלצת טקסט מכל קובץ, בלי קשר להפניות
    raw_text = extract_text_from_xml(file_path)
    
    if raw_text:
        combined_texts[base_date].append(raw_text)

print(f"סיום חילוץ. נמצאו {len(combined_texts)} תאריכים ייחודיים.")

# שמירת הקבצים המאוחדים
for date, text_list in combined_texts.items():
    final_combined_text = ' '.join(text_list)
    
    # שם הקובץ המאוחד: debates_YYYY-MM-DD.txt
    output_filename = f"debates_{date}.txt"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(final_combined_text)

print(f"הטקסטים המאוחדים (מבוססי <p>) נשמרו בהצלחה בספרייה **{OUTPUT_DIR}**")


# In[7]:


import nltk
nltk.download('punkt')


# In[8]:


import nltk
nltk.download('punkt_tab')


# In[10]:


import spacy
import os

# הגדרות נתיבים
INPUT_DIR = os.path.join('data', 'combined_xml_files')
OUTPUT_DIR = os.path.join('data', 'tokenized_text_spacy') # שם חדש
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. טעינת מודל השפה הקטן של spaCy
# נטען אותו פעם אחת מחוץ ללולאה לחיסכון בזמן
print("טוען את מודל השפה של spaCy...")
nlp = spacy.load("en_core_web_sm")
print("המודל נטען.")

print(f"מתחיל ניקוי סימני פיסוק (עם spaCy). קבצים יישמרו ב- {OUTPUT_DIR}")

# קריאה של כל הקבצים המאוחדים
for filename in os.listdir(INPUT_DIR):
    if filename.endswith('.txt'):
        input_path = os.path.join(INPUT_DIR, filename)
        
        with open(input_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
            
        # 2. עיבוד הטקסט עם spaCy
        # זו הדרך שבה spaCy מבצע Tokenization
        doc = nlp(raw_text)
        
        # 3. חילוץ הטקסט של כל טוקן
        # spaCy שומר כל טוקן כאובייקט, אנחנו ניקח רק את הטקסט שלו
        tokens = [token.text for token in doc]
        
        # 4. חיבור הטוקנים בחזרה למחרוזת עם רווחים
        cleaned_text = ' '.join(tokens)
        
        # 5. שמירת הקובץ הנקי
        output_path = os.path.join(OUTPUT_DIR, filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)

print("סיום ניקוי סימני פיסוק (spaCy).")


# In[ ]:


import os
import glob
import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from collections import Counter
import nltk

nltk.download("punkt")
nltk.download("stopwords")

STOP_WORDS = set(stopwords.words("english"))

def load_texts(folder_path):
    docs, filenames = [], []
    for file_path in glob.glob(os.path.join(folder_path, "*.txt")):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read().lower()
            tokens = [w for w in word_tokenize(text) if w.isalpha() and w not in STOP_WORDS]
            docs.append(tokens)
            filenames.append(os.path.basename(file_path))
    return docs, filenames

def filter_rare_words(docs, min_freq=5):
    freq = Counter([word for doc in docs for word in doc])
    filtered_docs = [[w for w in doc if freq[w] >= min_freq] for doc in docs]
    return filtered_docs

def save_json_vectors(docs, filenames, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    bm25 = BM25Okapi(docs)
    for doc_tokens, fname in zip(docs, filenames):
        vector = {}
        for term in bm25.idf.keys():
            f = doc_tokens.count(term)
            if f > 0:
                score = bm25.idf[term] * ((f * (bm25.k1 + 1)) / (f + bm25.k1 * (1 - bm25.b + bm25.b * len(doc_tokens) / bm25.avgdl)))
                vector[term] = round(score, 3)
        with open(os.path.join(output_dir, fname.replace(".txt", ".json")), "w", encoding="utf-8") as f:
            json.dump(vector, f, ensure_ascii=False, indent=2)

# === הפעלת התהליך ===
base_data = "data"
base_model = "models"

# עיבוד גרסת spaCy
spacy_docs, spacy_names = load_texts(os.path.join(base_data, "tokenized_text_spacy"))
spacy_docs = filter_rare_words(spacy_docs)
save_json_vectors(spacy_docs, spacy_names, os.path.join(base_model, "bm25_word_json_dict"))

# עיבוד גרסת Lemmatized
lemm_docs, lemm_names = load_texts(os.path.join(base_data, "lemmatized_text"))
lemm_docs = filter_rare_words(lemm_docs)
save_json_vectors(lemm_docs, lemm_names, os.path.join(base_model, "bm25_lemm_json_dict"))

print("✅ Done! All vectors saved per document.")


# In[4]:


import os
import glob
import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from collections import Counter
import nltk

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

STOP_WORDS = set(stopwords.words("english"))

def load_texts(folder_path):
    docs, filenames = [], []
    print(f"טוען ומנקה טקסטים מ: {folder_path}")
    for file_path in glob.glob(os.path.join(folder_path, "*.txt")):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read().lower()
            tokens = [w for w in word_tokenize(text) if w.isalpha() and w not in STOP_WORDS]
            docs.append(tokens)
            filenames.append(os.path.basename(file_path))
    return docs, filenames

def filter_rare_words(docs, min_freq=5):
    print("מסנן מילים נדירות...")
    freq = Counter([word for doc in docs for word in doc])
    # שומרים את אוצר המילים שאינו נדיר
    vocab = set(word for word, count in freq.items() if count >= min_freq)
    filtered_docs = [[w for w in doc if w in vocab] for doc in docs]
    print(f"גודל אוצר המילים המקורי: {len(freq)}, אחרי סינון: {len(vocab)}")
    return filtered_docs, vocab

def save_json_vectors_optimized(docs, filenames, output_dir, vocab):
    """
    גרסה מהירה: מחשבת BM25 רק עבור המילים שבאמת מופיעות בכל מסמך.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"בונה מודל BM25 Okapi עבור {output_dir}...")
    bm25 = BM25Okapi(docs)
    
    print("יוצר ושומר וקטורים (בשיטה המהירה)...")
    for doc_tokens, fname in zip(docs, filenames):
        vector = {}
        
        # --- 1. האופטימיזציה ---
        # סופרים את המילים רק פעם אחת עבור המסמך הנוכחי
        doc_freqs = Counter(doc_tokens)
        
        # --- 2. האופטימיזציה ---
        # רצים *רק* על המילים הייחודיות במסמך זה (לא על כל המילון)
        for term in doc_freqs.keys():
            # (אין צורך ב-if f > 0 כי אנחנו יודעים שהמילה קיימת)
            f = doc_freqs[term]
            
            # --- זהירות: `rank_bm25` לא מחזיר ציון למילים שלא ב-IDF ---
            # אבל אנחנו סיננו את `docs` עם `vocab` אז זה אמור להיות בסדר
            if term not in bm25.idf:
                continue # מילה זו הייתה נדירה מדי והוסרה
                
            # אותה נוסחה בדיוק מהקוד שלך
            score = bm25.idf[term] * ((f * (bm25.k1 + 1)) / (f + bm25.k1 * (1 - bm25.b + bm25.b * len(doc_tokens) / bm25.avgdl)))
            vector[term] = round(score, 3)
            
        # שמירת קובץ ה-JSON
        with open(os.path.join(output_dir, fname.replace(".txt", ".json")), "w", encoding="utf-8") as f:
            json.dump(vector, f, ensure_ascii=False, indent=2)

# === הפעלת התהליך (עם האופטימיזציה) ===
base_data = "data"
base_model = "models"

# --- עיבוד גרסת spaCy (Word) ---
spacy_docs_raw, spacy_names = load_texts(os.path.join(base_data, "tokenized_text_spacy"))
spacy_docs_filtered, spacy_vocab = filter_rare_words(spacy_docs_raw)
save_json_vectors_optimized(spacy_docs_filtered, spacy_names, 
                            os.path.join(base_model, "bm25_word_json_dict"), 
                            spacy_vocab)
print("--- סיום עיבוד Word ---")


# --- עיבוד גרסת Lemmatized (Lemm) ---
lemm_docs_raw, lemm_names = load_texts(os.path.join(base_data, "lemmatized_files"))
lemm_docs_filtered, lemm_vocab = filter_rare_words(lemm_docs_raw)
save_json_vectors_optimized(lemm_docs_filtered, lemm_names, 
                            os.path.join(base_model, "bm25_lemm_json_dict"), 
                            lemm_vocab)
print("--- סיום עיבוד Lemm ---")

print("✅ Done! All optimized vectors saved per document.")


# In[7]:


import os  # 🚀 הוספנו את זה
import glob
import json
import re
import sys
import subprocess
import importlib.util

# --- Auto-install gensim if not found ---
if importlib.util.find_spec("gensim") is None:
    print("Installing gensim ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gensim"])

# Now safe to import
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import nltk

# --- Downloads for NLTK ---
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
STOP_WORDS = set(stopwords.words("english"))

# --- Utility functions ---

def clean_text(text):
    """Remove punctuation, digits, quotes, etc."""
    text = re.sub(r"[^\w\s]", " ", text)  # remove punctuation
    text = re.sub(r"\d+", " ", text)      # remove numbers
    text = text.lower().strip()
    return text

def load_docs(folder, remove_stop=False):
    """Load all txt files and tokenize"""
    docs, names = [], []
    for path in glob.glob(os.path.join(folder, "*.txt")):
        with open(path, "r", encoding="utf-8") as f:
            text = clean_text(f.read())
            tokens = [w for w in word_tokenize(text) if w.isalpha()]
            if remove_stop:
                tokens = [w for w in tokens if w not in STOP_WORDS]
            docs.append(tokens)
            names.append(os.path.basename(path))
    return docs, names

def build_doc_vectors(docs, model):
    """Average all word vectors in each document"""
    vectors = []
    for tokens in docs:
        valid = [w for w in tokens if w in model.wv]
        if valid:
            vecs = np.array([model.wv[w] for w in valid])
            vectors.append(np.mean(vecs, axis=0))
        else:
            vectors.append(np.zeros(model.vector_size))
    return vectors

def save_vectors(vectors, names, out_dir):
    """Save each document vector as its own JSON"""
    os.makedirs(out_dir, exist_ok=True)
    for vec, name in zip(vectors, names):
        with open(os.path.join(out_dir, name.replace(".txt", ".json")), "w", encoding="utf-8") as f:
            json.dump({"vector": vec.tolist()}, f, ensure_ascii=False, indent=2)

# --- Main Configuration ---
base_data = "data"
base_model = "models"
VECTOR_SIZE = 300

# --- 🚀 אופטימיזציה: שימוש בכל הליבות הזמינות ---
CPU_CORES = os.cpu_count() or 1 # ( or 1 למקרה שהפקודה נכשלת)
print(f"Word2Vec | מפעיל אופטימיזציה: משתמש ב-{CPU_CORES} ליבות CPU.")


# --- 1️⃣ tokenized_text_spacy - with stopwords ---
print("\nמתחיל מודל 1/4: Word (עם stop-words)")
docs, names = load_docs(os.path.join(base_data, "tokenized_text_spacy"), remove_stop=False)
model = Word2Vec(sentences=docs, vector_size=VECTOR_SIZE, window=5, min_count=2, workers=CPU_CORES) # 🚀
save_vectors(build_doc_vectors(docs, model), names, os.path.join(base_model, "w2v_word_with_stop"))

# --- 2️⃣ tokenized_text_spacy - no stopwords ---
print("\nמתחיל מודל 2/4: Word (בלי stop-words)")
docs_ns, names_ns = load_docs(os.path.join(base_data, "tokenized_text_spacy"), remove_stop=True)
model_ns = Word2Vec(sentences=docs_ns, vector_size=VECTOR_SIZE, window=5, min_count=2, workers=CPU_CORES) # 🚀
save_vectors(build_doc_vectors(docs_ns, model_ns), names_ns, os.path.join(base_model, "w2v_word_no_stop"))

# --- 3️⃣ lemmatized_text - with stopwords ---
print("\nמתחיל מודל 3/4: Lemma (עם stop-words)")
lemm_docs, lemm_names = load_docs(os.path.join(base_data, "lemmatized_files"), remove_stop=False)
model_lemm = Word2Vec(sentences=lemm_docs, vector_size=VECTOR_SIZE, window=5, min_count=2, workers=CPU_CORES) # 🚀
save_vectors(build_doc_vectors(lemm_docs, model_lemm), lemm_names, os.path.join(base_model, "w2v_lemm_with_stop"))

# --- 4️⃣ lemmatized_text - no stopwords ---
print("\nמתחיל מודל 4/4: Lemma (בלי stop-words)")
lemm_docs_ns, lemm_names_ns = load_docs(os.path.join(base_data, "lemmatized_files"), remove_stop=True)
model_lemm_ns = Word2Vec(sentences=lemm_docs_ns, vector_size=VECTOR_SIZE, window=5, min_count=2, workers=CPU_CORES) # 🚀
save_vectors(build_doc_vectors(lemm_docs_ns, model_lemm_ns), lemm_names_ns, os.path.join(base_model, "w2v_lemm_no_stop"))

print("\n✅ Done! 4 folders with 300-dimensional Word2Vec document vectors created.")


# In[ ]:


import os
import glob
import json
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

print("--- התחלת סעיף ג': SimCSE ---")

# --- 1. הגדרת נתיבים ---
INPUT_DIR = os.path.join('data', 'combined_xml_files') 
OUTPUT_DIR = os.path.join('models', 'simcse_origen')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 2. בדיקת האצת GPU (MPS) עבור M1/M2 ---
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("זיהוי M1/M2 GPU (MPS). מפעיל האצה.")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("זיהוי NVIDIA GPU (CUDA). מפעיל האצה.")
else:
    device = torch.device("cpu")
    print("לא זוהתה האצת GPU. משתמש ב-CPU.")

# --- 3. טעינת מודל SimCSE ---
model_name = 'princeton-nlp/unsup-simcse-bert-base-uncased'
print(f"טוען את המודל {model_name}...")
model = SentenceTransformer(model_name, device=device)
print("המודל נטען.")

# --- 4. טעינת המסמכים ---
all_texts = []
all_names = []
print(f"טוען את כל קבצי המקור מ-{INPUT_DIR}...")
for file_path in glob.glob(os.path.join(INPUT_DIR, "*.txt")):
    with open(file_path, "r", encoding="utf-8") as f:
        all_texts.append(f.read())
    all_names.append(os.path.basename(file_path))
print(f"נמצאו {len(all_texts)} מסמכים.")

# --- 5. יצירת וקטורים (Embeddings) ---
print("מתחיל ביצירת וקטורי SimCSE...")
vectors = model.encode(
    all_texts, 
    show_progress_bar=True, 
    batch_size=32, # אפשר להגדיל אם יש לך הרבה VRAM
    convert_to_numpy=True 
)
print("יצירת הוקטורים הושלמה.")

# --- 6. שמירת הוקטורים ---
print(f"שומר את הוקטורים ב-{OUTPUT_DIR}...")
for vec, name in zip(vectors, all_names):
    output_path = os.path.join(OUTPUT_DIR, name.replace(".txt", ".json"))
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"vector": vec.tolist()}, f, ensure_ascii=False, indent=2)

print("✅ סיום סעיף ג'! וקטורי SimCSE נשמרו.")


# In[2]:


import os
import glob
import json
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

print("--- התחלת סעיף ד': SBERT ---")

# --- 1. הגדרת נתיבים ---
# שימוש באותם קבצי מקור כמו SimCSE
INPUT_DIR = os.path.join('data', 'combined_xml_files') 
# תיקיית פלט לפי שם הקבוצה במטלה
OUTPUT_DIR = os.path.join('models', 'sbert_origen')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 2. בדיקת האצת GPU (MPS) ---
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("זיהוי M1/M2 GPU (MPS). מפעיל האצה.")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("זיהוי NVIDIA GPU (CUDA). מפעיל האצה.")
else:
    device = torch.device("cpu")
    print("לא זוהתה האצת GPU. משתמש ב-CPU.")

# --- 3. טעינת מודל SBERT ---
# זהו מודל SBERT פופולרי ומומלץ לשימוש כללי
model_name = 'all-mpnet-base-v2'
print(f"טוען את המודל {model_name}...")
model = SentenceTransformer(model_name, device=device)
print("המודל נטען.")

# --- 4. טעינת המסמכים ---
# הקוד זהה לסעיף הקודם
all_texts = []
all_names = []
print(f"טוען את כל קבצי המקור מ-{INPUT_DIR}...")
for file_path in glob.glob(os.path.join(INPUT_DIR, "*.txt")):
    with open(file_path, "r", encoding="utf-8") as f:
        all_texts.append(f.read())
    all_names.append(os.path.basename(file_path))
print(f"נמצאו {len(all_texts)} מסמכים.")

# --- 5. יצירת וקטורים (Embeddings) ---
print("מתחיל ביצירת וקטורי SBERT...")
vectors = model.encode(
    all_texts, 
    show_progress_bar=True, 
    batch_size=32,
    convert_to_numpy=True 
)
print("יצירת הוקטורים הושלמה.")

# --- 6. שמירת הוקטורים ---
print(f"שומר את הוקטורים ב-{OUTPUT_DIR}...")
for vec, name in zip(vectors, all_names):
    output_path = os.path.join(OUTPUT_DIR, name.replace(".txt", ".json"))
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"vector": vec.tolist()}, f, ensure_ascii=False, indent=2)

print("✅ סיום סעיף ד'! וקטורי SBERT נשמרו.")


# In[9]:


import os
from nltk.tokenize import word_tokenize

# הגדרות נתיבים
INPUT_DIR = os.path.join('data', 'combined_xml_files')
OUTPUT_DIR = os.path.join('data', 'tokenized_text_nltk') # שם חדש כדי להבדיל
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"מתחיל ניקוי סימני פיסוק (עם NLTK). קבצים יישמרו ב- {OUTPUT_DIR}")

# קריאה של כל הקבצים המאוחדים
for filename in os.listdir(INPUT_DIR):
    if filename.endswith('.txt'):
        input_path = os.path.join(INPUT_DIR, filename)
        
        with open(input_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
            
        # 1. שימוש בטוקנייזר החכם של NLTK
        # הוא יודע לטפל ב-don't, U.S.A., וכו' בצורה נכונה
        tokens = word_tokenize(raw_text)
        
        # 2. חיבור הטוקנים בחזרה למחרוזת עם רווחים
        # התוצאה תהיה טקסט שבו סימני הפיסוק מופרדים כהלכה
        cleaned_text = ' '.join(tokens)
        
        # 3. שמירת הקובץ הנקי
        output_path = os.path.join(OUTPUT_DIR, filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)

print("סיום ניקוי סימני פיסוק (NLTK).")


# In[4]:


import os
import glob
import json
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import mutual_info_classif, chi2
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans 
import numpy as np

print("--- התחלת סעיף ה' (גישת K-Means משולבת) ---")

# --- 1. הגדרת נתיבים ---
# !!!
# !!! הרץ את התא פעם אחת עם הנתיב הזה:
TFIDF_DIR = os.path.join('models', 'bm25_word_json_dict')
# !!!
# !!! ואז שנה לנתיב הבא והרץ שוב:
# TFIDF_DIR = os.path.join('models', 'bm25_lemm_json_dict')
# !!!
print(f"מעבד את המטריצה: {TFIDF_DIR}")

# --- 2. טעינת הוקטורים (X) - פעם אחת ---
doc_vectors = [] # X
filenames = []   

print("טוען קבצי JSON...")
for file_path in glob.glob(os.path.join(TFIDF_DIR, "*.json")):
    fname = os.path.basename(file_path)
    with open(file_path, 'r', encoding='utf-8') as f:
        doc_vectors.append(json.load(f))
    filenames.append(fname)

print(f"נמצאו {len(doc_vectors)} מסמכים.")

# --- 3. בניית המטריצה (X) - פעם אחת ---
print("ממיר את רשימת ה-dicts למטריצת פיצ'רים (X)...")
vectorizer = DictVectorizer(sparse=True)
X_sparse = vectorizer.fit_transform(doc_vectors)
feature_names = vectorizer.get_feature_names_out()

print(f"המטריצה נוצרה בגודל: {X_sparse.shape} (מסמכים, מאפיינים)")

# --- 4. 💡 לולאה על 3 ערכי K שונים ---
# שנה את הרשימה הזו כרצונך
K_VALUES = [5, 10, 15] 

for k in K_VALUES:
    print("\n" + "="*60)
    print(f"--- מתחיל ניתוח עבור K = {k} ---")
    print("="*60)

    # --- 4a. יצירת תוויות (y) באמצעות K-Means ---
    print(f"מבצע קיבוץ (clustering) ל- {k} קבוצות...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    y = kmeans.fit_predict(X_sparse) # 🚀 זוהי ה-y החדשה שלנו!
    
    print(f"התפלגות המסמכים בקבוצות (K={k}):")
    print(np.bincount(y))

    # --- 5. חישוב המדדים (עם התוויות החדשות מ-K-Means) ---

    # מדד 1: Information Gain
    print(f"[K={k}] מחשב Information Gain...")
    print(f" [K={k}] (ממיר למטריצה צפופה)...")
    X_dense = X_sparse.toarray()
    ig_scores = mutual_info_classif(X_dense, y, discrete_features=False)
    ig_results = pd.DataFrame({'feature': feature_names, 'info_gain': ig_scores})
    ig_results = ig_results.sort_values(by='info_gain', ascending=False)

    # מדד 2: Chi-squared
    print(f"[K={k}] מחשב Chi-squared...")
    chi2_scores, p_values = chi2(X_sparse, y)
    chi2_results = pd.DataFrame({'feature': feature_names, 'chi2_score': chi2_scores, 'p_value': p_values})
    chi2_results = chi2_results.sort_values(by='chi2_score', ascending=False)

    # מדד 3: Gini Impurity
    print(f"[K={k}] מחשב Gini Impurity...")
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_sparse, y)
    gini_scores = clf.feature_importances_
    gini_results = pd.DataFrame({'feature': feature_names, 'gini_importance': gini_scores})
    gini_results = gini_results.sort_values(by='gini_importance', ascending=False)

    print(f"[K={k}] החישובים הסתיימו.")

    # --- 6. שמירת התוצאות לאקסל ---
    excel_filename = f"feature_analysis_KMeans_k={k}_{os.path.basename(TFIDF_DIR)}.xlsx"
    print(f"שומר תוצאות לקובץ: {excel_filename}")

    with pd.ExcelWriter(excel_filename) as writer:
        ig_results.to_excel(writer, sheet_name='Information Gain', index=False)
        chi2_results.to_excel(writer, sheet_name='Chi-squared', index=False)
        gini_results.to_excel(writer, sheet_name='Gini Importance', index=False)
    
    print(f"--- Top 10 תוצאות עבור K={k} (להשוואה מהירה) ---")
    print(f"\n--- Top 10 Info Gain (K={k}) ---")
    print(ig_results.head(10))
    print(f"\n--- Top 10 Chi-squared (K={k}) ---")
    print(chi2_results.head(10))
    print(f"\n--- Top 10 Gini Importance (K={k}) ---")
    print(gini_results.head(10))

print("\n\n✅✅✅ סיום! כל ניתוחי ה-K-Means הושלמו. ✅✅✅")


# In[11]:


import os
import glob
import json
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import mutual_info_classif, chi2
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans 
import numpy as np
import matplotlib.pyplot as plt

print("--- התחלת סעיף ה' (גישת K-Means, קובץ מאוחד) ---")

# --- 1. הגדרת נתיבים ---
# !!!
# !!! הרץ את התא פעם אחת עם הנתיב הזה:
TFIDF_DIR = os.path.join('models', 'bm25_word_json_dict')
# !!!
# !!! ואז שנה לנתיב הבא והרץ שוב:
# TFIDF_DIR = os.path.join('models', 'bm25_lemm_json_dict')
# !!!
print(f"מעבד את המטריצה: {TFIDF_DIR}")

# --- 2. טעינת הוקטורים (X) - פעם אחת ---
doc_vectors = [] # X
filenames = []   

print("טוען קבצי JSON...")
for file_path in glob.glob(os.path.join(TFIDF_DIR, "*.json")):
    fname = os.path.basename(file_path)
    with open(file_path, 'r', encoding='utf-8') as f:
        doc_vectors.append(json.load(f))
    filenames.append(fname)
print(f"נמצאו {len(doc_vectors)} מסמכים.")

# --- 3. בניית המטריצה (X) - פעם אחת ---
print("ממיר את רשימת ה-dicts למטריצת פיצ'רים (X)...")
vectorizer = DictVectorizer(sparse=True)
X_sparse = vectorizer.fit_transform(doc_vectors)
feature_names = vectorizer.get_feature_names_out()
print(f"המטריצה נוצרה בגודל: {X_sparse.shape} (מסמכים, מאפיינים)")

# --- 4. 💡 הכנת הטבלאות המאוחדות ---
# ניצור DataFrames ריקים עם המילים כשורות (אינדקס)
ig_results_all = pd.DataFrame(index=feature_names)
chi2_results_all = pd.DataFrame(index=feature_names)
gini_results_all = pd.DataFrame(index=feature_names)

K_VALUES = [5, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200] 
max_ig_scores = []
max_chi2_scores = []

# --- 5. 💡 לולאה על 3 ערכי K שונים ---
for k in K_VALUES:
    print("\n" + "="*60)
    print(f"--- מתחיל ניתוח עבור K = {k} ---")
    print("="*60)

    # --- 5a. יצירת תוויות (y) באמצעות K-Means ---
    print(f"מבצע קיבוץ (clustering) ל- {k} קבוצות...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    y = kmeans.fit_predict(X_sparse) 
    print(f"התפלגות המסמכים בקבוצות (K={k}): {np.bincount(y)}")

    # --- 5b. חישוב המדדים ---

    # מדד 1: Information Gain
    print(f"[K={k}] מחשב Information Gain...")
    X_dense = X_sparse.toarray()
    ig_scores = mutual_info_classif(X_dense, y, discrete_features=False)
    max_ig_scores.append(np.max(ig_scores))
    # 💡 הוספת התוצאות כעמודה חדשה
    ig_results_all[f'info_gain_k={k}'] = ig_scores

    # מדד 2: Chi-squared
    print(f"[K={k}] מחשב Chi-squared...")
    chi2_scores, p_values = chi2(X_sparse, y)
    max_chi2_scores.append(np.max(chi2_scores))
    # 💡 הוספת התוצאות כעמודות חדשות
    chi2_results_all[f'chi2_score_k={k}'] = chi2_scores
    chi2_results_all[f'p_value_k={k}'] = p_values

    # מדד 3: Gini Impurity
    print(f"[K={k}] מחשב Gini Impurity...")
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_sparse, y)
    gini_scores = clf.feature_importances_
    # 💡 הוספת התוצאות כעמודה חדשה
    gini_results_all[f'gini_importance_k={k}'] = gini_scores

    print(f"[K={k}] החישובים הסתיימו.")

    # --- הדפסת Top 10 זמנית ---
    print(f"\n--- Top 10 Info Gain (K={k}) ---")
    print(ig_results_all[[f'info_gain_k={k}']].sort_values(by=f'info_gain_k={k}', ascending=False).head(10))

print("\n\n✅✅✅ כל חישובי ה-K-Means הושלמו. ✅✅✅")

# --- 6. שמירת הקובץ המאוחד ---
excel_filename = f"feature_analysis_KMeans_CONSOLIDATED_{os.path.basename(TFIDF_DIR)}.xlsx"
print(f"שומר את כל התוצאות לקובץ אקסל אחד: {excel_filename}")

# נמיין כל טבלה לפי התוצאות של K=10 (או K הראשון ברשימה)
k_to_sort_by = K_VALUES[1] if len(K_VALUES) > 1 else K_VALUES[0]

with pd.ExcelWriter(excel_filename) as writer:
    ig_results_all.sort_values(by=f'info_gain_k={k_to_sort_by}', ascending=False).to_excel(writer, sheet_name='Information Gain')
    chi2_results_all.sort_values(by=f'chi2_score_k={k_to_sort_by}', ascending=False).to_excel(writer, sheet_name='Chi-squared')
    gini_results_all.sort_values(by=f'gini_importance_k={k_to_sort_by}', ascending=False).to_excel(writer, sheet_name='Gini Importance')

print("✅ סיום! הקובץ המאוחד נשמר בהצלחה.")

# --- 7. יצירת גרפים ---
plt.figure(figsize=(12, 6))
plt.plot(K_VALUES, max_ig_scores, marker='o')
plt.title('Max Information Gain vs. K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Max Information Gain')
plt.grid(True)
plt.savefig('info_gain_vs_k.png')
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(K_VALUES, max_chi2_scores, marker='o')
plt.title('Max Chi-squared vs. K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Max Chi-squared Score')
plt.grid(True)
plt.savefig('chi2_vs_k.png')
plt.show()

print("✅ גרפים נשמרו בהצלחה.")



# In[11]:





# In[ ]:




