# MethoOntoChat

## Run in the cloud

### Instructions to run Streamlit app in the cloud

1) Open your web browser.

2) Go to: [https://methoontochat.streamlit.app/](https://methoontochat.streamlit.app/)

3) Note that the files used in the app are from the GitHub repository.

---

## Run locally

### Instructions to run Streamlit app locally

1) Assuming you are using conda:
```bash
conda activate methoontochat
```

2) Install the dependencies:
```bash
pip install -r requirements.txt
```

3) Add the necessary keys in `.streamlit/secrets.toml`. The keys are:
  * pinecone_api_secret
  * hugging_face_api_secret
  * langchain_api_secret

4) Run the app:
```bash
streamlit run methoontochat_streamlit_app.py --server.port 70
```

### Instructions to preprocess files for RAG
Note: It is assumed that the required dependencies are installed.

1) Place your PDF corpus in a folder within `data/input`.

2) Open the `preprocessing_corpus_files.py` script and specify the folder containing the files you want to preprocess.

3) While running Docker execute the following command to use the image `lfoppiano/grobid:0.8.0`:
```bash
docker run --rm --init --ulimit core=0 -p 8070:8070 lfoppiano/grobid:0.8.0
```

4) Execute the `preprocessing_corpus_files.py` script, specifying the folder to process.

### Instructions to upload preprocessed files to Pinecone

1) Ensure your preprocessed files are stored in a Google Drive repository.

2) Open the `creating_index_pinecone.py` script.

3) Introduce your Pinecone key in the script.

4) This script is set up to run in Google Colab, so make sure to mount your Google Drive.

5) Execute the script to upload the preprocessed files to the Pinecone database.
