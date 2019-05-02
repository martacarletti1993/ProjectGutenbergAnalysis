To run the code you will need to have the following installed:

* Python 3
* Python Libraries:
	* requests
	* bs4
	* nltk
	* unidecode
	* wikipedia
	* pymongo
	* urllib
	* SPARQLWrapper
* MongoDB

To run the code:

1. Ensure that the `author.py` and `dataExtraction.py` files are in the same folder
2. Start your MongoDB server. By default, it should be running at localhost/27017. If this is different in your system, change the address and the port on lines 17 and 18 in CarlettiPavlova_UE803.py
3. In your terminalNavigate to the folder where you have saved the files and type `python3 dataExtraction.py`
4. The system will start collecting the information for each author for whom it finds more than k sentences throughout their books. The following information is stored in MongoDB:
	1. Name of the author
	2. URL to the English Wikipedia of the author (if found)
	3. Multilingual abstracts for the author (if any)
	4. Literary movements the author belongs to (if any)
	5. The names of the books downloaded and the path to each book
5. The data (books per author) is saved in ./lab1_data
6. The program keeps track of the saved authors and their books during runtime. This is stored in the `books_by_author` dictionary
7. You can change the number of `k` value for sentences retrieved by author on line 244 when calling the `process_author` function 

Note: We have submitted a JSON file (lab1.json) containing the information collected in Step 4 with `k` = 10,000.

database_structure.png shows a sample author entry in our database. We have used Robomongo for viewing the results.

The database allows for querying various types of information. Some sample queries can be found in `mongo_queries.py`. You may need to change the MongoHost and MongoPort on lines 3 and 4 in that file.

Concerning books per author, the database contains only the relative paths to each book and not the text. Due to the size of the data, we have not included it in the submission folder, but all the downloaded files can be found here: https://drive.google.com/drive/folders/17h5uCFrTkQv-5P4uJIa2ak9mRrYoBOg1?usp=sharing
