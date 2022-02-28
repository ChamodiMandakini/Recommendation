# Import libraries
#!pip install rake_nltk
from rake_nltk import Rake   # ensure this is installed

import nltk
nltk.download('all')
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import warnings
warnings.filterwarnings("ignore")


import os

from flask import Flask
from flask import request, url_for, redirect, render_template, jsonify


app = Flask(__name__)

@app.route('/')
def home():
    return render_template("questions.html")

@app.route('/questions',methods=['POST'])
def questions():
    
    int_features = [x for x in request.form.values()]
    print(int_features)

    #Sentence Tokenization
    from nltk.tokenize import sent_tokenize
    
    text = int_features[5]
    """text="Reboot was an initiative by us to improve the knowledge of the GCE O/L students on the key subjects of Science, Mathematics and English. We identified that due to the prevailing situation around the world, this year’s O/L students were in a difficult place because they were unable to attend to their studies in a regular manner. We, therefore decided to have a series of online quizzes and webinars through which we could reach out to students around the country, encouraging and motivating them to learn these subjects in an entertaining manner. The project was conducted under three main phases;Phase 1- This phase was conducted under three sessions; Science, Mathematics and English. Each session spanned a period of three days. On day 1, we gave the participants a question paper which covered the subject material and encouraged them to try it out on their own. On day 2, we held a webinar and our own university undergraduates taught and discussed that paper with the participants via Zoom. On day 3, we had a quiz for all participants using google forms through which we selected the highest scoring participants for phase 2.Phase 2- Only 50 participants were chosen to move forward into phase 2. We had another quiz within phase 2, which consisted of all three subjects covered in Phase 1 using the Kahoot platform. We also had a Motivational session conducted by Dr. Tharindu Weerasinghe - Senior Lecturer, University of Kelaniya via Zoom which was open to anyone interested and not just for the O/L students who had registered for the project.Phase 3- We chose only the top 10 scorers of the quiz conducted in Phase 2 for the Grand Finale. Under Phase 3, we had a quiz for the top 10 participants which consisted of General Knowledge and IQ questions. We conducted this final event via Zoom and live streamed it on Facebook as well. We also gave the audience an opportunity to win exciting gifts by having a General Knowledge quiz for them as well. Mr. Thanura Madugeeth and Mr. Suneera Sumanga facilitated the event for us with entertainment in-between the quizzes and the formalities All participants were given e-certificates while the top 10 participants and winners received gift vouchers."
"""


    tokenized_text=sent_tokenize(text)
    #print(tokenized_text)



    #Word Tokenization
    from nltk.tokenize import word_tokenize
    tokenized_word=word_tokenize(text)
    #print(tokenized_word)



    #Stopwords
    from nltk.corpus import stopwords
    stop_words=set(stopwords.words("english"))
    #print(stop_words)



    #Removing Stopwords
    filtered_sent=[]
    for w in tokenized_word:
        if w not in stop_words:
            filtered_sent.append(w)

    #print("Filterd Sentence:",filtered_sent)



    '''#Stemming
    from nltk.stem import PorterStemmer
    from nltk.tokenize import sent_tokenize, word_tokenize

    ps = PorterStemmer()
    stemmed_words=[]

    for sw in filtered_sent:
        stemmed_words.append(ps.stem(sw))
    print("Stemmed Sentence:",stemmed_words)

    '''

    #Lexicon Normalization
    #performing stemming and Lemmatization
    final_words =[]
    from nltk.stem.wordnet import WordNetLemmatizer
    lem = WordNetLemmatizer()

    for lw in filtered_sent:
        lword = lem.lemmatize(lw,"v")
        final_words.append(lword)
    #print("Lemmatized Word:", final_words)


    goal_1=0
    goal_2=0
    goal_3=0
    goal_4=0
    goal_5=0
    goal_6=0
    goal_7=0
    goal_8=0
    goal_9=0
    goal_10=0
    goal_11=0
    goal_12=0
    goal_13=0
    goal_14=0
    goal_15=0
    goal_16=0
    goal_17=0






    #df = pd.read_csv('https://query.data.world/s/uikepcpffyo2nhig52xxeevdialfl7')   # 250 rows × 38 columns
    df = pd.read_csv('data.csv', encoding = "ISO-8859-1")   # same data 17 rows × 4 columns
     
    
    # to remove punctuations from Keywords
    df['Keywords'] = df['Keywords'].str.replace('[^\,\w\s]','')
    df['Description'] = df['Description'].str.replace('[^\,\w\s]','')
    # # alternative way to remove punctuations, same result

    
    
    # to extract key words from Plot to a list
    df['Key_words'] = ''   # initializing a new column
    r = Rake()   # use Rake to discard stop words (based on english stopwords from NLTK)

    
    
    for index, row in df.iterrows():
        r.extract_keywords_from_text(row['Keywords'])   # to extract key words from Plot, default in lower case
        key_words_dict_scores = r.get_word_degrees()    # to get dictionary with key words and their scores
        row['Key_words'] = list(key_words_dict_scores.keys())   # to assign list of key words to new column


        
        
    # to extract key words from Plot to a list
    df['New_Description'] = ''   # initializing a new column
    r = Rake()   # use Rake to discard stop words (based on english stopwords from NLTK)

    
    
    for index, row in df.iterrows():
        r.extract_keywords_from_text(row['Description'])   # to extract key words from Plot, default in lower case
        key_words_dict_scores = r.get_word_degrees()    # to get dictionary with key words and their scores
        row['New_Description'] = list(key_words_dict_scores.keys())   # to assign list of key words to new column

        
        
    # to combine 4 lists (4 columns) of key words into 1 sentence under Bag_of_words column
    df['Bag_of_words'] = ''
    columns = ['Key_words', 'New_Description']

    
    
    for index, row in df.iterrows():
        words = ''
        for col in columns:
            words += ' '.join(row[col]) + ' '
        row['Bag_of_words'] = words

        
        
    # strip white spaces infront and behind, replace multiple whitespaces (if any)
    df['Bag_of_words'] = df['Bag_of_words'].str.strip().str.replace('   ', ' ').str.replace('  ', ' ')

    
    
    df = df[['SDG','Bag_of_words']]

    
    
    #create vector representation for Bag_of_words and the similarity matrix
    # to generate the count matrix
    count = CountVectorizer()
    count_matrix = count.fit_transform(df['Bag_of_words'])


    # to generate the cosine similarity matrix (size 17 x 17)
    # rows represent all movies; columns represent all movies
    # cosine similarity: similarity = cos(angle) = range from 0 (different) to 1 (similar)
    # all the numbers on the diagonal are 1 because every movie is identical to itself (cosine value is 1 means exactly identical)
    # matrix is also symmetrical because the similarity between A and B is the same as the similarity between B and A.
    # for other values eg 0.1578947, movie x and movie y has similarity value of 0.1578947

    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    print(cosine_sim)


    
    # to create a Series for movie titles which can be used as indices (each index is mapped to a movie title)
    indices = pd.Series(df['SDG'])

    #run and test the recommender model


    # this function takes in a movie title as input and returns the top 10 recommended (similar) movies

    def recommend(title, cosine_sim = cosine_sim):
        recommended_SDG = []
        idx = indices[indices == title].index[0]   # to get the index of the movie title matching the input movie
        score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False)   # similarity scores in descending order

        top_10_indices = list(score_series.iloc[0:5].index)   # to get the indices of top 10 most similar movies

        # [1:11] to exclude 0 (index 0 is the input movie itself)

        for i in top_10_indices:   # to append the titles of top 10 similar movies to the recommended_movies list
            recommended_SDG.append(list(df['SDG'])[i])

        return recommended_SDG



    for words in final_words:
        if words=="poverty" or words=="income" or  words=="poor" or  words=="rural" or  words=="money" or  words=="recovery" or  words=="fund" or  words=="poorness" or  words=="misery" or  words=="gutter" or  words=="wealth":
            goal_1 = goal_1+1
            

        elif words=="hunger" or words=="food" or  words=="production" or  words=="agriculture" or  words=="starvation" or  words=="feed" or  words=="nutritions" or  words=="cook" or  words=="prepare" or  words=="meal" or  words=="eat" or  words=="donate" or  words=="distribute" or  words=="lunch":
            goal_2 = goal_2+1

        elif words=="healthy" or words=="health" or  words=="disease" or  words=="mental" or words=="well-being" or words=="deaths" or  words=="fitness" or  words=="physical" or  words=="food" or  words=="wellness" or  words=="healthiness" or  words=="active" or  words=="illness" or  words=="sick" or  words=="sickness" or  words=="condition" or  words=="disorder":
            goal_3 = goal_3+1

        elif words=="education" or words=="literacy" or  words=="scholarships" or  words=="primary" or words=="knowledge" or words=="student" or words=="subject" or words=="school" or words=="quiz" or  words=="training" or  words=="train" or  words=="teach" or  words=="teaching" or  words=="tution" or  words=="higher" or  words=="degree" or  words=="learn" or  words=="learning" or  words=="reading" or  words=="books" or  words=="children" or  words=="grades":
            goal_4 = goal_4+1

        elif words=="equal" or words=="empower" or  words=="gender" or  words=="rights" or  words=="discrimination" or  words=="women" or  words=="men" or  words=="domestic" or  words=="equity" or  words=="pay" or  words=="equality" or  words=="sex" or  words=="sexual" or  words=="feminism" or  words=="girls":
            goal_5 = goal_5+1

        elif words=="clean" or words=="water" or  words=="hygiene" or  words=="sanitation" or  words=="fresh" or  words=="pure" or  words=="sanitary" or  words=="wash" or  words=="natural" or  words=="river" or  words=="resources" or words=="unpolluted" or  words=="pollute" or  words=="pollution" or  words=="drinking" or  words=="purified" or  words=="purify" or  words=="drinkable":
            goal_6 = goal_6+1

        elif words=="energy" or words=="electricity" or  words=="heat" or  words=="transport" or  words=="nuclear" or  words=="fuel" or  words=="magnetic" or  words=="power" or  words=="electric" or  words=="gas" or  words=="solar" or  words=="panel" or  words=="sun" or  words=="tropical" or  words=="transform" or  words=="battery" or  words=="chemical" or  words=="turbine" or  words=="generator" or  words=="generate" or  words=="greenhouse" or  words=="cell" or  words=="engine" or  words=="transport" or  words=="transpotation" or  words=="steam" :
            goal_7 = goal_7+1

        elif words=="productivity" or words=="tourism" or  words=="finance" or  words=="employment" or  words=="product" or  words=="work" or  words=="workers" or  words=="increse" or words=="opportunity" or  words=="development" or  words=="economy" or  words=="economic" or  words=="labour" or  words=="factory" or  words=="occupation" or  words=="growth" or  words=="ingrowth" or  words=="production" or  words=="automation" or  words=="market" or  words=="rates" or  words=="rate" or  words=="consumer" or  words=="build" or  words=="income":
            goal_8 = goal_8+1

        elif words=="industry" or words=="innovation" or  words=="manufacturing" or  words=="workers" or words=="industrialization" or  words=="discover" or  words=="new" or  words=="idea" or  words=="ideas" or  words=="thinking" or  words=="rights" or  words=="inspiration" or  words=="innovative" or  words=="clever" or  words=="creative" or  words=="original" or  words=="invention" or  words=="design":
            goal_9 = goal_9+1

        elif words=="inequality" or words=="opportinities" or  words=="income" or  words=="discrimination" or  words=="imbalance" or  words=="divergence"  or  words=="gender"  or  words=="social"  or  words=="unequal":
            goal_10 = goal_10+1

        elif words=="community" or words=="safe" or  words=="culture" or  words=="urban" or  words=="lifestyle" or  words=="cities" or  words=="city"  or  words=="village"  or  words=="family"  or  words=="communities"  or  words=="service"  or  words=="affordable"  or  words=="comfortable"  or  words=="transport"  or  words=="shops"  or  words=="stores" or  words=="children" or words=="public":
            goal_11 = goal_11+1

        elif words=="production" or words=="consumption" or  words=="recycle" or  words=="consumer"  or  words=="sustainable"  or  words=="resposible"  or  words=="response" or  words=="sustainability" :
            goal_12 = goal_12+1

        elif words=="climate" or words=="disaster" or  words=="wildfire" or  words=="natural"  or  words=="weather"  or  words=="atmosphere"  or  words=="tempreture"  or  words=="rain"  or  words=="flood"  or  words=="humidity"  or  words=="warming"  or  words=="greenhouse" or  words=="rainfall"  or  words=="change"  or  words=="wind"  or  words=="thunderstorm" or  words=="predict":
            goal_13 = goal_13+1

        elif words=="marine" or words=="sea" or words=="seas" or  words=="ocean" or words=="oceans" or  words=="water"  or  words=="algae"  or  words=="bacteria" or words=="bacterias"  or  words=="coral" or  words=="whales"  or  words=="corals" or  words=="fish"  or  words=="jellyfish"  or  words=="whale"  or  words=="octopus" or words=='overfishing' or words== 'fishing':
            goal_14 = goal_14+1

        elif words=="ecosystem" or words=="land" or  words=="biodiversity" or  words=="species" or words=="deforestation" or  words=="trees"  or  words=="plants"  or  words=="environment"  or  words=="animals" or  words=="creatures"  or  words=="forest"  or  words=="forests" or  words=="elephants" :
            goal_15 = goal_15+1

        elif words=="law" or words=="crime" or  words=="terrorism" or  words=="freedom" or  words=="fairness"  or  words=="fair" or  words=="justice"  or  words=="honesty"  or  words=="integrity":
            goal_16 = goal_16+1

        elif words=="partnership" or words=="stakeholders" or  words=="cooperation" or  words=="international" or  words=="country" or  words=="trade" or  words=="sector" or  words=="export" or  words=="exports":
            goal_17 = goal_17+1

    integers = [goal_1, goal_2, goal_3, goal_4, goal_5, goal_6, goal_7, goal_8, goal_9, goal_10, goal_11, goal_12, goal_13, goal_14, goal_15, goal_16, goal_17]
    
    
    print(integers)

    largest_integer = max(integers) 
    integers.remove(largest_integer)
    second_largest_integer = max(integers) 
    integers.remove(second_largest_integer)
    third_largest_integer = max(integers) 
    integers.remove(third_largest_integer)
    fourth_largest_integer = max(integers) 
    integers.remove(fourth_largest_integer)
    fifth_largest_integer = max(integers) 
    integers.remove(fifth_largest_integer)
    sixth_largest_integer = max(integers)
    
    print(largest_integer, second_largest_integer, third_largest_integer, fourth_largest_integer, fifth_largest_integer, sixth_largest_integer)

    result = ''
    
    if largest_integer == second_largest_integer:
            second_largest_integer = third_largest_integer
            print ('1==2',second_largest_integer)
       
    if second_largest_integer == third_largest_integer:
            third_largest_integer = fourth_largest_integer
            print ('2==3',third_largest_integer)
            
    if third_largest_integer == fourth_largest_integer:
            fourth_largest_integer = fifth_largest_integer   
            print ('3==4',fourth_largest_integer)
            
    if fourth_largest_integer == fifth_largest_integer:
            fifth_largest_integer = sixth_largest_integer   
             
            
            
    print(largest_integer, second_largest_integer, third_largest_integer, fourth_largest_integer)
    
    
    
    if int_features[2] == "Australia" or int_features[2] == "Norway" or int_features[2] == "Ireland" or int_features[2] == "Switzerland" or int_features[2] == "Iceland" or int_features[2] == "Hong Kong" or int_features[2] == "Germany" or int_features[2] == "Sweden" or int_features[2] == "Netherlands" or int_features[2] == "Netherland Antilles" or int_features[2] == "Denmark" or int_features[2] == "Finland" or int_features[2] == "Singapore" or int_features[2] == "United Kingdom" or int_features[2] == "New Zealand" or int_features[2] == "Belgium" or int_features[2] == "United States of America" or int_features[2] == "China":
        
        if largest_integer == goal_1:
            largest_integer = second_largest_integer
            second_largest_integer = third_largest_integer
            print (largest_integer)
            
        if largest_integer == goal_2:
            largest_integer = second_largest_integer
            second_largest_integer = third_largest_integer
            print (largest_integer)
            
        if second_largest_integer == goal_1:
            second_largest_integer = third_largest_integer
            third_largest_integer = fourth_largest_integer
            print (second_largest_integer)
            
        if second_largest_integer == goal_2:
            second_largest_integer = fourth_largest_integer
            third_largest_integer = fourth_largest_integer
            print (second_largest_integer)
    
    print(largest_integer, second_largest_integer)
    
    if largest_integer == goal_1:
        result = 'Goal 1 - No poverty'
        print("Most recommended SDG is Goal 1 - No poverty")  
        r = 1
        
    elif largest_integer == goal_2:
        print("Most recommended SDG is Goal 2 - Zero hunger") 
        result = 'Goal 2 - Zero hunger'
        r = 2
        
    elif largest_integer == goal_3:
        print("Most recommended SDG is Goal 3 - Good health and well-being")
        result = 'Goal 3 - Good health and well-being'
        r = 3
        
    elif largest_integer == goal_4:
        print("Most recommended SDG is Goal 4 - Quality education")
        result = 'Goal 4 - Quality education'
        r = 4
        
    elif largest_integer == goal_5:
        print("Most recommended SDG is Goal 5 - Gender equality")
        result = 'Goal 5 - Gender equality'
        r = 5
        
    elif largest_integer == goal_6:
        print("Most recommended SDG is Goal 6 - Clean water and sanitation") 
        result = 'Goal 6 - Clean water and sanitation'
        r = 6
        
    elif largest_integer == goal_7:
        print("Most recommended SDG is Goal 7 - Affordable and clean energy")
        result = 'Goal 7 - Affordable and clean energy'
        r = 7
        
    elif largest_integer == goal_8:
        print("Most recommended SDG is Goal 8 - Decent work and economic growth")
        result = 'Goal 8 - Decent work and economic growth'
        r = 8
        
    elif largest_integer == goal_9:
        print("Most recommended SDG is Goal 9 - Industry, Innovation and Infrastructure")
        result = 'Goal 9 - Industry, Innovation and Infrastructure'
        r = 9
        
    elif largest_integer == goal_10:
        print("Most recommended SDG is Goal 10 - Reduced inequality")
        result = 'Goal 10 - Reduced inequality'
        r = 10
        
    elif largest_integer == goal_11:
        print("Most recommended SDG is Goal 11 - Sustainable cities and communities")
        result = 'Goal 11 - Sustainable cities and communities'
        r = 11
        
    elif largest_integer == goal_12:
        print("Most recommended SDG is Goal 12 - Responsible consumption and production")
        result = 'Goal 12 - Responsible consumption and production'
        r = 12
        
    elif largest_integer == goal_13:
        print("Most recommended SDG is Goal 13 - Climate action")
        result = 'Goal 13 - Climate action'
        r = 13
        
    elif largest_integer == goal_14:
        print("Most recommended SDG is Goal 14 - Life below water")
        result = 'Goal 14 - Life below water'
        r = 14
        
    elif largest_integer == goal_15:
        print("Most recommended SDG is Goal 15 - Life on land")
        result = 'Goal 15 - Life on land'
        r = 15
        
    elif largest_integer == goal_16:
        print("Most recommended SDG is Goal 16 - Peace, justice and strong institutions")
        result = 'Goal 16 - Peace, justice and strong institutions'
        r = 16
        
    elif largest_integer == goal_17:
        print("Most recommended SDG is Goal 17 - Partnership for the goals")
        result = 'Goal 17 - Partnership for the goals'
        r = 17
    
    
    ### second recommended goal###########################
    if second_largest_integer == goal_1:
        result2 = 'Goal 1 - No poverty'  
        r2 = 1
        
    elif second_largest_integer == goal_2:
        result2 = 'Goal 2 - Zero hunger'
        r2 = 2
        
    elif second_largest_integer == goal_3:
        result2 = 'Goal 3 - Good health and well-being'
        r2 = 3
        
    elif second_largest_integer == goal_4:
        result2 = 'Goal 4 - Quality education'
        r2 = 4
        
    elif second_largest_integer == goal_5:
        result2 = 'Goal 5 - Gender equality'
        r2 = 5
        
    elif second_largest_integer == goal_6:
        result2 = 'Goal 6 - Clean water and sanitation'
        r2 = 6
        
    elif second_largest_integer == goal_7:
        result2 = 'Goal 7 - Affordable and clean energy'
        r2 = 7
        
    elif second_largest_integer == goal_8:
        result2 = 'Goal 8 - Decent work and economic growth'
        r2 = 8
        
    elif second_largest_integer == goal_9:
        result2 = 'Goal 9 - Industry, Innovation and Infrastructure'
        r2 = 9
        
    elif second_largest_integer == goal_10:
        result2 = 'Goal 10 - Reduced inequality'
        r2 = 10
        
    elif second_largest_integer == goal_11:
        result2 = 'Goal 11 - Sustainable cities and communities'
        r2 = 11
        
    elif second_largest_integer == goal_12:
        result2 = 'Goal 12 - Responsible consumption and production'
        r2 = 12
        
    elif second_largest_integer == goal_13:
        result2 = 'Goal 13 - Climate action'
        r2 = 13
        
    elif second_largest_integer == goal_14:
        result2 = 'Goal 14 - Life below water'
        r2 = 14
        
    elif second_largest_integer == goal_15:
        result2 = 'Goal 15 - Life on land'
        r2 = 15
        
    elif second_largest_integer == goal_16:
        result2 = 'Goal 16 - Peace, justice and strong institutions'
        r2 = 16
        
    elif second_largest_integer == goal_17:
        result2 = 'Goal 17 - Partnership for the goals'
        r2 = 17
       
    return render_template('recommend.html', sdg1 = result, sdg2= result2, pic1=r, pic2=r2)
    

if __name__ == '__main__':
    app.run(host = "0.0.0.0", port = 80, debug=True)
