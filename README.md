# LangChain_project
This project is used to learn how Langchain work. This will be a End to End project using Langchain

## Compare Pdf
This script is compares two paper in pdf format. The idea is to pass in the location to the pdf using command line arguments. The file will then return a read comparison for both papers.

### Example command to run 
After activating the venv for project and going to the correct directory. You can run

```
py .\compare_pdfs.py --doc1 .\data\doc_1.pdf --doc2 .\data\doc_2.pdf
```

## Summaises web page
This file will take in a url using command line arguements and summarise the webpage in basic information. 

### Example command to run 
After activating the venv for project and going to the correct directory. You can run
```
py .\summaries_web_page.py --webpage https://www.bbc.co.uk/news
```

## Create Resturant Menu
This file will create a resturant name and menu based on the provided cusine.
```
py .\resturant_project.py --cuisine indian
```

## RAG exmaple
This script builds a Retrieval-Augmented Generation (RAG) pipeline that combines information from a PDF document and a YouTube video transcript. It uses:
* LangChain to load and chunk the PDF,
* YouTubeTranscriptApi to fetch video transcripts,
* HuggingFace sentence-transformers embeddings with FAISS vector store for document indexing,
* Ollama LLM (llama3.1) to answer questions based on retrieved documents.

The pipeline allows you to ask a question that summarizes the key ideas from both the PDF and the YouTube video transcript.

### How to run
In a terminal 

```
ollama serve
ollama pull llama3.1
```
```
python your_script.py --doc path/to/document.pdf --video YOUTUBE_VIDEO_ID
```

Video ID can be found in the URL of the youtube video.

#### Example command
For https://www.youtube.com/watch?v=iGJ1XSkCyU0&ab_channel=BorisMeinardus -> video id is after "watch?v" 

```
py .\ragExample.py --doc .\data\doc_1.pdf --video iGJ1XSkCyU0
```