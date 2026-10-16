import streamlit as st
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
import argparse
import time
import os
import subprocess
import fcntl
import signal
import tempfile
import pandas as pd



load_dotenv()

# Set default values from .env
os.environ["TOKENIZERS_PARALLELISM"] = "false"
model_n_ctx = int(os.environ.get('MODEL_N_CTX', 1000))
model_n_batch = int(os.environ.get('MODEL_N_BATCH', 8))
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS', 4))
model_type = os.environ.get('MODEL_TYPE', 'GPT4All')
model_path = os.environ.get('MODEL_PATH', 'LLM/ggml-gpt4all-j-v1.3-groovy.bin')
embeddings_model_name = os.environ.get('EMBEDDINGS_MODEL_NAME', 'all-MiniLM-L6-v2')
persist_directory = os.environ.get('PERSIST_DIRECTORY')
# Set up the sidebar


from constants import CHROMA_SETTINGS

os.makedirs("source_documents", exist_ok=True)


#st.set_page_config(page_title="𝚑𝚊𝚌𝚔🅶🅿🆃", page_icon="https://raw.githubusercontent.com/NoDataFound/hackGPT/main/res/hackgpt_fav.png", layout="wide")
# Define the chat history data as a Pandas DataFrame

CSS = """
img {
    box-shadow: 0px 10px 15px rgba(0, 0, 0, 0.2);
}
"""

st.markdown(f'<style>{CSS}</style>', unsafe_allow_html=True)
st.sidebar.image('https://raw.githubusercontent.com/NoDataFound/hackGPT/main/res/hackGPT_logo.png', width=300)
github_logo = "https://raw.githubusercontent.com/NoDataFound/hackGPT/main/res/github.png"
hackGPT_repo = "https://github.com/NoDataFound/hackGPT"

st.sidebar.markdown(f"[![GitHub]({github_logo})]({hackGPT_repo} 'hackGPT repo')")
st.sidebar.title("File Upload")
st.image('https://raw.githubusercontent.com/NoDataFound/hackGPT/main/res/hackGPT_logo.png', width=800)    
embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})

uploaded_files = [os.path.join("source_documents", filename) for filename in os.listdir("source_documents") if filename != ".DS_Store"]
uploaded_file = st.sidebar.file_uploader("Choose a file", type=["csv","docx","doc","enex","eml","epub","html","md","msg","odt","pdf","pptx ","ppt ","txt"])

SOURCE_DIR = "source_documents"
# Admission limits protecting the shared corpus and ingestion resources.
MAX_UPLOAD_BYTES = int(os.environ.get('MAX_UPLOAD_BYTES', 10 * 1024 * 1024))
MAX_CORPUS_BYTES = int(os.environ.get('MAX_CORPUS_BYTES', 200 * 1024 * 1024))
MAX_CORPUS_FILES = int(os.environ.get('MAX_CORPUS_FILES', 200))
INGEST_TIMEOUT_SECONDS = int(os.environ.get('INGEST_TIMEOUT_SECONDS', 600))
INGEST_COOLDOWN_SECONDS = int(os.environ.get('INGEST_COOLDOWN_SECONDS', 30))
INGEST_LOCK_PATH = os.path.join(tempfile.gettempdir(), "hackgptp_ingest.lock")


def corpus_usage(exclude_path=None):
    """Return (file_count, total_bytes) for the shared source directory."""
    count, total = 0, 0
    for name in os.listdir(SOURCE_DIR):
        path = os.path.join(SOURCE_DIR, name)
        if name == ".DS_Store" or not os.path.isfile(path):
            continue
        if exclude_path is not None and os.path.abspath(path) == os.path.abspath(exclude_path):
            continue
        count += 1
        total += os.path.getsize(path)
    return count, total


def run_ingest(file_path):
    """Run ingest.py under a process-wide lock, a cooldown and a hard deadline.

    Returns (ok, message).
    """
    now = time.time()
    last = st.session_state.get("last_ingest_time", 0)
    if now - last < INGEST_COOLDOWN_SECONDS:
        return False, "Ingestion rate limit reached. Please wait before submitting another ingestion request."

    lock_file = open(INGEST_LOCK_PATH, "a")
    try:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return False, "Another ingestion is already in progress. Please try again later."

        st.session_state["last_ingest_time"] = now
        proc = subprocess.Popen(["python3", "ingest.py", file_path], start_new_session=True)
        try:
            proc.communicate(timeout=INGEST_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                proc.kill()
            proc.wait()
            return False, "Ingestion exceeded the permitted processing time and was terminated."
        return True, ""
    finally:
        lock_file.close()  # releases the flock


def save_uploaded_file(uploaded_file):
    # Strip any directory components from the client-supplied name.
    file_name = os.path.basename(uploaded_file.name.replace("\\", "/"))
    if not file_name or file_name in (".", ".."):
        return None, "Invalid file name."
    file_path = os.path.join(SOURCE_DIR, file_name)

    size = uploaded_file.size
    if size > MAX_UPLOAD_BYTES:
        return None, f"File exceeds the maximum upload size of {MAX_UPLOAD_BYTES} bytes."
    count, total = corpus_usage(exclude_path=file_path)
    if count + 1 > MAX_CORPUS_FILES or total + size > MAX_CORPUS_BYTES:
        return None, "The shared document corpus has reached its size limit. Delete files before uploading more."

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Call ingest.py script (guarded by lock, cooldown and timeout)
    ok, message = run_ingest(file_path)
    if not ok:
        return file_path, message

    return file_path, ""



def split_text_into_chunks(text, chunk_size):
    chunks = []
    while len(text) > chunk_size:
        chunks.append(text[:chunk_size])
        text = text[chunk_size:]
    if text:
        chunks.append(text)
    return chunks
 
if uploaded_file is not None:
    # Streamlit reruns the script on every interaction; only process each upload once.
    upload_key = (uploaded_file.name, uploaded_file.size)
    if st.session_state.get("processed_upload") != upload_key:
        file_path, error_message = save_uploaded_file(uploaded_file)
        st.session_state["processed_upload"] = upload_key
        if file_path is None:
            st.sidebar.error(error_message)
        elif error_message:
            st.sidebar.warning("File saved but not ingested: " + error_message)
        else:
            st.sidebar.success("File uploaded successfully.")
            uploaded_files.append(file_path)
uploaded_files = [os.path.join("source_documents", filename) for filename in os.listdir("source_documents") if filename != ".DS_Store"]

df_data = []

total_words = 0  # Variable to store the total word count

for idx, document_file in enumerate(uploaded_files):
    file_name = os.path.basename(document_file)
    file_type = os.path.splitext(file_name)[1].lstrip('.')
    date_trained = os.path.getmtime(document_file)
    word_count = 0
    sample = ""

    if file_type.lower() != "pdf":  # Skip line reading for PDF files
        with open(document_file, "r") as f:
            lines = f.readlines()
            if len(lines) > 0:
                word_count = sum(len(line.split()) for line in lines)  # Count words in each line
                sample = lines[0].strip()
        
            total_words += word_count  # Add current document's word count to the total

    df_data.append({
        'File Name': file_name,
        'File Type': file_type,
        'Date Trained': pd.to_datetime(date_trained, unit='s').strftime('%m-%d-%y'),
        'Word Count': word_count,
        'Sample': sample
    })

df = pd.DataFrame(df_data)

# Sidebar options
st.sidebar.title("Training Data")
show_training_data = st.sidebar.checkbox("Show Training Data")
selected_files = st.sidebar.multiselect("Select Files to Re-process", uploaded_files)
delete_training_data = st.sidebar.button("Delete Selected Files")
reprocess_training_data = st.sidebar.button("Re-process Selected Files")

if delete_training_data:
    # Delete selected files logic here
    for file_path in selected_files:
        os.remove(file_path)
    st.sidebar.success("Selected files deleted.")
    st.stop()

if reprocess_training_data:
    # Reprocess selected files logic here
    # ingest.py processes the whole source directory regardless of the argument,
    # so a single guarded run is sufficient for any number of selected files.
    if selected_files:
        ok, message = run_ingest(selected_files[0])
        if ok:
            st.sidebar.success("Selected files re-processed.")
        else:
            st.sidebar.error(message)
    st.stop()

if show_training_data:
    st.info("Training Data")
    st.dataframe(df.style.set_properties(subset=['Date Trained'], **{'font-size': '12px'}))

def main():
    # Load the embeddings model
    args = parse_arguments()
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]
    
    if model_type == "LlamaCpp":
        llm = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, n_batch=model_n_batch, callbacks=callbacks, verbose=False)
    elif model_type == "GPT4All":
        llm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', n_batch=model_n_batch, callbacks=callbacks, verbose=False)
    else:
        raise Exception(f"Model type {model_type} is not supported. Please choose one of the following: LlamaCpp, GPT4All")
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=not args.hide_source)
    
    query = st.text_input("", value="Ask your question", key="question_input")
    submit_button = st.button("Submit")

    if submit_button:
        st.spinner("Processing Question")
        start = time.time()
        res = qa(query)
        answer, docs = res['result'], [] if args.hide_source else res['source_documents']
        end = time.time()
        
        st.code(f"> Answer (took {round(end - start, 2)} s.):")
        st.success(answer)

        for document in docs:
            st.code("'Answer derived from "+ document.metadata["source"]+ " in this section: ") 
            st.info(document.page_content)


def parse_arguments():
    parser = argparse.ArgumentParser(description='privateGPT: Ask questions to your documents without an internet connection, '
                                                 'using the power of LLMs.')
    parser.add_argument("--hide-source", "-S", action='store_true',
                        help='Use this flag to disable printing of source documents used for answers.')

    parser.add_argument("--mute-stream", "-M",
                        action='store_true',
                        help='Use this flag to disable the streaming StdOut callback for LLMs.')

    return parser.parse_args()

if __name__ == "__main__":
    main()
