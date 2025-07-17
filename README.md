# LangChain_project
This project is used to learn how Langchain work. This will be a End to End project using Langchain

## Compare Pdf
This script is compares two paper in pdf format. The idea is to pass in the location to the pdf using command line arguments. The file will then return a read comparison for both papers.

### Example command to run 
After activating the venv for project and going to the correct directory. You can run

```py .\compare_pdfs.py --doc1 .\data\doc_1.pdf --doc2 .\data\doc_2.pdf```

## Summaises web page
This file will take in a url using command line arguements and summarise the webpage in basic information. 

### Example command to run 
After activating the venv for project and going to the correct directory. You can run
```py .\summaries_web_page.py --webpage https://www.bbc.co.uk/news```

## Create Resturant Menu
This file will create a resturant name and menu based on the provided cusine.
```py .\resturant_project.py --cuisine indian```