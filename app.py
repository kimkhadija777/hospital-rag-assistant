import os
import re
import json
import zipfile
import shutil

import numpy as np
import faiss
import gdown
import streamlit as st

from sentence_transformers import SentenceTransformer
from groq import Groq


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Hospital RAG Assistant",
    page_icon="🏥",
    layout="wide"
)


# ============================================================
# CONFIGURATION
# ============================================================

DRIVE_URL = (
    "https://drive.google.com/file/d/"
    "1Y4u5C9_4U6X5FNv8WqvITkC6fEy0j7n4/view?usp=drivesdk"
)

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

GROQ_MODEL = "openai/gpt-oss-120b"

DOWNLOAD_FILE = "/tmp/hospital_knowledge_base.zip"

EXTRACT_ROOT = "/tmp/hospital_rag_data"


# ============================================================
# HEADER
# ============================================================

st.title("🏥 Hospital RAG Knowledge Base Assistant")

st.write(
    "Ask questions about hospital admissions, departments, "
    "emergency services, hospital information, and patient safety."
)

st.info(
    "This is an educational RAG demonstration using a "
    "fictional hospital knowledge base. It is not medical advice."
)


# ============================================================
# GROQ API KEY FROM STREAMLIT SECRETS
# ============================================================

try:

    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

except Exception:

    st.error(
        "❌ GROQ_API_KEY is missing from Streamlit Secrets."
    )

    st.stop()


# ============================================================
# DOWNLOAD KNOWLEDGE BASE
# ============================================================

@st.cache_resource
def download_and_extract_kb():

    if os.path.exists(EXTRACT_ROOT):

        shutil.rmtree(EXTRACT_ROOT)

    os.makedirs(
        EXTRACT_ROOT,
        exist_ok=True
    )

    if os.path.exists(DOWNLOAD_FILE):

        os.remove(DOWNLOAD_FILE)

    match = re.search(
        r"/d/([a-zA-Z0-9_-]+)",
        DRIVE_URL
    )

    if not match:

        raise ValueError(
            "Could not extract Google Drive file ID."
        )

    file_id = match.group(1)

    gdown.download(
        id=file_id,
        output=DOWNLOAD_FILE,
        quiet=True
    )

    if not os.path.exists(DOWNLOAD_FILE):

        raise FileNotFoundError(
            "Knowledge base ZIP could not be downloaded."
        )

    with zipfile.ZipFile(
        DOWNLOAD_FILE,
        "r"
    ) as zip_ref:

        zip_ref.extractall(
            EXTRACT_ROOT
        )

    return EXTRACT_ROOT


# ============================================================
# LOAD DOCUMENTS
# ============================================================

@st.cache_resource
def load_documents():

    root = download_and_extract_kb()

    txt_files = []

    for current_root, dirs, files in os.walk(root):

        for filename in files:

            if filename.lower().endswith(".txt"):

                txt_files.append(
                    os.path.join(
                        current_root,
                        filename
                    )
                )

    txt_files = sorted(txt_files)

    if not txt_files:

        raise ValueError(
            "No TXT documents found."
        )

    documents = []

    for file_path in txt_files:

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as f:

            text = f.read().strip()

        relative_path = os.path.relpath(
            file_path,
            root
        )

        parts = relative_path.split(
            os.sep
        )

        category = (
            parts[0]
            if len(parts) >= 2
            else "general"
        )

        documents.append({

            "source":
                os.path.basename(file_path),

            "path":
                relative_path,

            "category":
                category,

            "text":
                text
        })

    return documents


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    text = text.replace(
        "\r\n",
        "\n"
    )

    text = text.replace(
        "\r",
        "\n"
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


# ============================================================
# CHUNKING
# ============================================================

def create_chunks(
    text,
    chunk_size=850,
    overlap=120
):

    text = clean_text(text)

    paragraphs = [
        p.strip()
        for p in re.split(
            r"\n\s*\n",
            text
        )
        if p.strip()
    ]

    chunks = []

    current = ""

    for paragraph in paragraphs:

        if (
            len(current)
            + len(paragraph)
            + 2
            <= chunk_size
        ):

            if current:

                current += (
                    "\n\n"
                    + paragraph
                )

            else:

                current = paragraph

        else:

            if current:

                chunks.append(
                    current.strip()
                )

            if (
                overlap > 0
                and current
            ):

                overlap_text = (
                    current[-overlap:]
                )

                first_space = (
                    overlap_text.find(" ")
                )

                if first_space != -1:

                    overlap_text = (
                        overlap_text[
                            first_space + 1:
                        ]
                    )

                current = (
                    overlap_text
                    + "\n\n"
                    + paragraph
                )

            else:

                current = paragraph

    if current.strip():

        chunks.append(
            current.strip()
        )

    return chunks


# ============================================================
# CREATE CHUNK RECORDS
# ============================================================

@st.cache_resource
def create_chunk_records():

    documents = load_documents()

    records = []

    chunk_id = 0

    for document in documents:

        chunks = create_chunks(
            document["text"],
            chunk_size=850,
            overlap=120
        )

        for local_id, chunk in enumerate(
            chunks
        ):

            records.append({

                "chunk_id":
                    chunk_id,

                "document_chunk_id":
                    local_id,

                "text":
                    chunk,

                "source":
                    document["source"],

                "path":
                    document["path"],

                "category":
                    document["category"]
            })

            chunk_id += 1

    return records


# ============================================================
# EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        EMBEDDING_MODEL
    )


# ============================================================
# BUILD FAISS INDEX
# ============================================================

@st.cache_resource
def build_faiss_index():

    records = create_chunk_records()

    model = load_embedding_model()

    texts = [
        item["text"]
        for item in records
    ]

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True
    )

    embeddings = embeddings.astype(
        "float32"
    )

    faiss.normalize_L2(
        embeddings
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )

    return index, records


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_documents(
    query,
    index,
    records,
    model,
    top_k=5
):

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    ).astype(
        "float32"
    )

    faiss.normalize_L2(
        query_embedding
    )

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for score, index_id in zip(
        scores[0],
        indices[0]
    ):

        if index_id == -1:

            continue

        item = records[
            int(index_id)
        ].copy()

        item["score"] = float(
            score
        )

        results.append(item)

    return results


# ============================================================
# GROQ RAG ANSWER
# ============================================================

def generate_answer(
    question,
    retrieved,
    client
):

    context_parts = []

    for number, item in enumerate(
        retrieved,
        start=1
    ):

        context_parts.append(
            f"""
SOURCE {number}

Category:
{item['category']}

File:
{item['source']}

Path:
{item['path']}

Content:
{item['text']}
"""
        )

    context = "\n".join(
        context_parts
    )

    system_prompt = """
You are a Hospital Knowledge Base Assistant.

Answer ONLY using the supplied knowledge-base context.

Rules:

1. Do not invent information.

2. If information is missing, say:
"I could not find that information in the
hospital knowledge base."

3. Do not invent phone numbers, addresses,
prices, doctors, policies, or emergency contacts.

4. Do not diagnose medical conditions.

5. Do not prescribe medicines or treatments.

6. For emergencies, advise the user to contact
local emergency services or a qualified medical
professional.

7. Keep answers clear and concise.

8. Mention relevant source files when useful.
"""

    user_prompt = f"""
QUESTION:
{question}

KNOWLEDGE BASE CONTEXT:
{context}
"""

    response = client.chat.completions.create(

        model=GROQ_MODEL,

        messages=[

            {
                "role":
                    "system",

                "content":
                    system_prompt
            },

            {
                "role":
                    "user",

                "content":
                    user_prompt
            }
        ],

        temperature=0.1,

        max_tokens=700
    )

    return (
        response
        .choices[0]
        .message
        .content
    )


# ============================================================
# LOAD RAG SYSTEM
# ============================================================

with st.spinner(
    "🔄 Loading hospital knowledge base and AI model..."
):

    try:

        faiss_index, chunk_records = (
            build_faiss_index()
        )

        embedding_model = (
            load_embedding_model()
        )

        groq_client = Groq(
            api_key=GROQ_API_KEY
        )

    except Exception as e:

        st.error(
            f"❌ Application startup error: {e}"
        )

        st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🏥 Knowledge Base")

    st.success(
        "RAG system is ready."
    )

    st.write(
        f"📄 Documents: "
        f"{len(load_documents())}"
    )

    st.write(
        f"🧩 Chunks: "
        f"{len(chunk_records)}"
    )

    st.write(
        f"🔢 Vectors: "
        f"{faiss_index.ntotal}"
    )

    st.divider()

    st.caption(
        "Educational demonstration only."
    )


# ============================================================
# USER QUESTION
# ============================================================

st.subheader(
    "💬 Ask the Hospital Assistant"
)

question = st.text_input(
    "Enter your question:",
    placeholder=(
        "Example: What documents are required "
        "for hospital admission?"
    )
)


# ============================================================
# ASK BUTTON
# ============================================================

if st.button(
    "🔎 Ask Assistant",
    type="primary"
):

    if not question.strip():

        st.warning(
            "Please enter a question first."
        )

    else:

        with st.spinner(
            "🔍 Searching knowledge base..."
        ):

            retrieved = retrieve_documents(
                question,
                faiss_index,
                chunk_records,
                embedding_model,
                top_k=5
            )

        if not retrieved:

            st.warning(
                "No relevant information was found."
            )

        else:

            with st.spinner(
                "🤖 Generating answer..."
            ):

                try:

                    answer = generate_answer(
                        question,
                        retrieved,
                        groq_client
                    )

                    st.subheader(
                        "🤖 Answer"
                    )

                    st.write(
                        answer
                    )

                except Exception as e:

                    st.error(
                        f"❌ Groq error: {e}"
                    )

                    answer = None

            st.subheader(
                "📚 Retrieved Sources"
            )

            for number, item in enumerate(
                retrieved,
                start=1
            ):

                with st.expander(
                    f"{number}. "
                    f"{item['source']} "
                    f"— {item['category']}"
                ):

                    st.write(
                        f"**Similarity score:** "
                        f"{item['score']:.4f}"
                    )

                    st.write(
                        f"**Path:** "
                        f"{item['path']}"
                    )

                    st.write(
                        item["text"]
                    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🏥 Hospital RAG Knowledge Base Assistant "
    "• Educational project"
  )
