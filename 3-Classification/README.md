To run the code you will need to have the following installed:

* Python 3
* Python Libraries:
	* nltk
	* spacy
	* wordcloud
	* matplotlib
	* pymongo
	* numpy
	* sklearn
* MongoDB

To run the code:

1. Start your MongoDB server. By default, it should be running at localhost/27017. If this is different in your system, change the address and the port on lines 15 and 16 in CarlettiPavlova_UE803_2.py
2. In your terminal, navigate to the folder where you have saved the files and type `python3 classification.py`
3. The system will call the `classification()` function, performing classification using Naive Bayes and Linear Regression on 4 features and then display the results. The files for the classification are stored in `lab3_data`. Namely, these features are:
	1. Tokens
	2. POS tags
	3. NER tags
	4. Average sentence length
4. If you wish to use different parameters for instance number, authors and sentences per instance, you can run the following functions and their respective parameters:
	`instances, labels = preprocess_data(instance_number, authors_number, sentences_per_instance)`
	`extract_features(instances, labels)`


We have run the system on input 1000 instances altogether, 20 authors and 10 sentences per instance. The files for that are located in `lab3_data/1000`. Along with these files, there are images obtained via matplotlib which show the results from the `best_alpha_figures()` which runs Naive Bayes on different alphas and visualises the result. Based on this, we have chosen to use alpha=0.1 for tokens and alpha=0.5 for the other 3 features.
