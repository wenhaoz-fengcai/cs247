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
	* Remove stop words in tokenized words
	* Tokenize the documents using **nltk** `Name Entity Tokenizer` package.
	
- Search.py
	* Python class which gets a list of tuples as input
	* Recursively search for each entity in the dbpedia
	* Output 2 lists, the new entities, and the existing ones
