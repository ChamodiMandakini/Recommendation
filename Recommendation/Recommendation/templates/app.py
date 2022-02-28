from flask import Flask
from flask import request, url_for, redirect, render_template, jsonify
from pycaret.regression import *
import pickle

# Import Dependencies 
import pandas as pd
import numpy as np

# Import summarize from gensim
from gensim.summarization.summarizer import summarize
from gensim.summarization import keywords# Import the library
# to convert MSword doc to txt for processing.
import docx2txt
import pathlib

import PyPDF2
import os

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import spacy 
#spacy.cli.download("en_core_web_sm")
spacy.load('en_core_web_sm')

import nltk
from nltk.corpus import stopwords
'''nltk.download('stopwords')
nltk.download('punkt') # one time execution
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')
nltk.download('wordnet')
nltk.download('brown')
nltk.download('maxent_ne_chunker')'''

from resume_parser import resumeparse
from docx import Document
import magic

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("home.html")



#get the pdfs and analyze-------------------------------------------------------------------------------------------------------------

@app.route('/analyze',methods=['POST'])
def analyze():
    
    pdf_path = request.files.get("job_description")
    
    print(pdf_path)
    #print('************************************________________________________________')
    uploaded_files = request.files.getlist("file[]")
    print (uploaded_files)
    
    # Store the resume in a variable
    '''description = docx2txt.process(pdf_path[0])
    text_description = str(description)#Summarize the text with ratio 0.1 (10% of the total words.)
    print (text_description)
    #summarize(text_description, ratio=0.99)'''
    
    paths = []
    matchPercentages = []
    email = []
    phone = []
    name = []
    total_exp = []
    university = []
    designition = []
    skills = []
    i = "1"
   
    # Store the resume in a variable
    description = docx2txt.process(pdf_path)
    text_description = str(description)#Summarize the text with ratio 0.1 (10% of the total words.)
    #summarize(text_description, ratio=0.99)
    print(text_description)

    #path = ""
    #pdf_files = os.listdir(uploaded_files)

    for file in uploaded_files:
        #absfile = os.path.join(path, file)
        print(file)
        #print('************************************________________________________________')

        # creating a pdf file object
        #pdfFileObj = open(absfile, 'rb')

        # creating a pdf reader object
        pdfReader = PyPDF2.PdfFileReader(file)

        # creating a page object
        pageObj = pdfReader.getPage(0)

        # extracting text from page
        text = pageObj.extractText()

        # Store the resume in a variable
        #resume = docx2txt.process('cv.docx')
        text_resume = str(text)#Summarize the text with ratio 0.1 (10% of the total words.)
        #summary = summarize(text_resume, ratio=0.1)
        #print(summary)
        #print('\n')

        
        document = Document()
        p = document.add_paragraph(text_resume)
        document.save('resume/'+i+'.docx')


        #using resume-parser extraxt the data
        data = resumeparse.read_file('resume/'+i+'.docx')
        #print(data)
        i = int(i)
        i = i+1
        i = str(i)
        # recycle the text variable from summarizing
        # creating A list of text
        text_list = [text_resume, text_description]



        cv = CountVectorizer()
        count_matrix = cv.fit_transform(text_list)


        # get the match percentage
        matchPercentage = cosine_similarity(count_matrix)[0][1] * 100

        matchPercentage = round(matchPercentage, 2) # round to two decimal
        #print('Your resume matches about '+ str(matchPercentage)+ '% of the job description.')

        paths.append(file)
        matchPercentages.append(matchPercentage)

        email.append(data['email'])
        phone.append(data['phone'])
        name.append(data['name'])
        total_exp.append(data['total_exp'])
        university.append(data['university'])
        skills.append(data['skills'])
        #emails.append(data['Companies'])

        # closing the pdf file object
        #pdfFileObj.close()



        

    paths = np.array(paths).flatten().tolist()
    name = np.array(name).flatten().tolist()
    email = np.array(email).flatten().tolist()
    university = np.array(university).flatten().tolist()
    total_exp =np.array(total_exp).flatten().tolist()
    skills = np.array(skills).flatten().tolist()
    matchPercentages = np.array(matchPercentages).flatten().tolist()

    
    #Create Dictionary of above variables
    cv_analyzer = {'Path': paths, 'Name': name, 'Email': email, 'University': university, 'Experience':total_exp, 'Skills':skills, 'Match Pecentage': matchPercentages
                 }

    #Convert Dictionary to data frame----------------------------------------------------------------------------------------
    cv_analyzer_df = pd.DataFrame(cv_analyzer)

    cv_analyzer_df = cv_analyzer_df.sort_values('Match Pecentage', ascending=False)

    # Export Dataframe to CSV
    cv_analyzer_df.to_csv("templates/cv_analyzer.csv", index=False, header=True)
    

    # SAVE CSV TO HTML USING PANDAS------------------------------------------------------------------------------------------
    cv_analyzer_csv = 'templates/cv_analyzer.csv'

    cv_analyzer_html = cv_analyzer_csv[:-3]+'html'


    df = pd.read_csv(cv_analyzer_csv, sep=',')
    df.to_html(cv_analyzer_html)


    #return render_template('result.html',pred='Summary :- {}'.format(cv_analyzer_df))
    return render_template('cv_analyzer.html')



'''@app.route('/result',methods=['POST'])
def result():
    
    return render_template('cv_analyzer.html')'''


if __name__ == '__main__':
    app.run(debug=True)
