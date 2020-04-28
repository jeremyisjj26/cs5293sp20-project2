Here is an example command to a function that will rank and cluster a news articles that had been uploaded from our local machine.
python3 article_parse.py

First I downloaded a zipfile from the Allen Institute for AI through Kaggl per class project.
        https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge.

You will need to register a account to access the data.

Once downloaded to my local machine, I uploaded it into a json_data directory to organize the input data.

I also created a test_json folder in which i tested a small number of files with the code because of time constraints.

I then imported libraires: {json,csv,glob}

I then Created several functions(x):
        outpuFile =  which is used to output the converted json file into a csv file labeled: "ConvertedData.csv"

        outputWriter =  which would write the csv file, and later write rows into the column labeled tex, from the key on the imported json files.

        data_folder  = which listed the location of the json data files("json_data").

I used the glob fuction to extract all files that had a .json extension in the filename within directory json_data.

        -i then created A for loop:
                - to bring in the data, assign and open the imported file
                - create a function to pass the new sourceFile into the json loader
                - and then another for loop was created to:
                        - query the json file down to only the body_text key
                        - create a new now array for that element
                        - and then antother for loop was created to:
                                - that will append the newRows from the key body_text in json_data
                                - write the new rows to the new outputfile

                                - This exported the output file, in the current directory, labeled "ConvertedData.csv"

I created a function df = that would call on pandas to bring the converted csv file into a pandas dataframe.

A fuction was ran to oupute the Total Rows in the new data selection: for my dataset.

I then trimmed the selection down to only the "text" colunm within the dataframe.

then ran several fuctions from: {nltk,re,numpy}

 - created a stop_words function that would fix an words within the list to its proper english interpretation
        - created functions to normalize the text within the document:
        - technique to make all the txt lower case
        - technique to take out any trailing or leading white space
        - tokenize the words
        - return the first line to check that the data was normalized
        - then created a function from numpy that would vectorize the normalized document
        - and then a fuction that would output a list of the normalized document for only the column labeled text, which reprented the contents of the articles that were uploaded.
I tested both the TfidfVectorizer and the CountVectorizer tools, but only output the clusters used from the CountVectorizer tool.

I then used the KMeans clustering algorithm to create clusters based on the similarity scores of the articles and bin them in 6 clusters.

I then used this output to interpret what the popular words were within each cluster.

from the collections import Counter function, i was able to store and group the clusters.

Once I labeld the clusters, i then sorted them by kmeans_cluster value and output statistics based on those clusters to output.txt.

 this included the top 9 feature names from each cluster using the cv.get_feature name function.

Due to time constraints, the output is for a minimal amount of files imported, but nonetheless i was able to get results.

This tells us that most of the documents are clustered in one main category, being that all of them are about COVID19, cluster 3. see output.txt file for results.
