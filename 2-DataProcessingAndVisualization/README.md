To run the code you will need to have the following installed:

* Python 3
* Python Libraries:
	* nltk
	* spacy
	* wordcloud
	* matplotlib
	* pymongo
	* numpy
* MongoDB

To run the code:

1. Start your MongoDB server. By default, it should be running at localhost/27017. If this is different in your system, change the address and the port on lines 15 and 16 in CarlettiPavlova_UE803_labsession2.py
2. In your terminalNavigate to the folder where you have saved the files and type `python3 dataExtraction.py`
3. The system will start collecting the information for each n sentences per author gathered in CarlettiPavlova_UE803.py. You can change to another `n` on line 304. The location of the files prepared for the following information is stored in MongoDB:
	1. Text segmented into sentences
	2. Tokenized text
	3. Each token with its relative POS
	4. Syntactic parsing of the sentences
4. The data (processed_files per author per n sentences) is saved in ./lab2_data


Concerning the information processed per author, the database contains only the relative paths to each processed file and not the text. Due to the size of the data, we have not included it in the submission folder, but the processed files for n=100 can be found under `lab2_data` here: https://drive.google.com/drive/folders/17h5uCFrTkQv-5P4uJIa2ak9mRrYoBOg1?usp=sharing

We have also uploaded the updated JSON file obtained from the database to the Google Drive folder.


We have also written a few functions for visualization. Namely:
	1. Comparison for vocabulary size for all authors
	2. Lemmatized and non-lemmatized word clouds for a selected author
	3. Box plots on sentence size for a selected author
	4. Bar plots for POS tag frequency by author

They can be tested using the examples in the main function.

All the functions except from the first one, save the images to a folder. Again, due to the size of the data, we have added some sample image on the google drive under the `images` folder.
