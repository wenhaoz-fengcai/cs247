- stanford-ner 
	* A Folder mainly consists of classifier data objects
	* Everything needed for English NER
	* Requires Java v1.8+
	* Please dont change/delete anything in this folder

- tests
	* Folder contains uni-tests cases

- Playground.ipynb
	* Code playground. Try different APIs.
	* Not part of our final code base

- Reader.py
	* Python class which reads the all files from given path under `data` folder
	* Reads files recursively
	* Tokenize the documents using **nltk** `Name Entity Tokenizer` package.