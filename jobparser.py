from pymongo import MongoClient, UpdateOne
from pprint import pprint
import nltk
# nltk.download('punkt') # uncomment on first run to get download
# nltk.download('stopwords') # uncomment on first run to get download
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from fastai.tabular.all import *
import pandas as pd
import os

from job_word_lists import *

CONNECTION_STRING = "mongodb://%s:%s@%s:27017/%s" % (os.environ["MONGO_USER"],
    os.environ["MONGO_PASS"], os.environ["MONGO_HOST"], os.environ["MONGO_DB_NAME"])
STOP_WORDS = stopwords.words('english')
IGNORED_WORDS = IGNOREABLE_WORDS + STOP_WORDS
SKILLS = set(GOOD_SKILLS + NEUTRAL_SKILLS + BAD_SKILLS)

def occurrence_cutoff(phrase, phrase_count, jobs_count):
    if phrase_count == 1:
        return False

    match len(phrase.split()):
        case 1: # 1%
            return phrase_count / jobs_count >= 0.01
        case 2:
            return phrase_count / jobs_count >= 0.01
        case 3: # 0.5%
            return phrase_count / jobs_count >= 0.005
        case 4:
            return phrase_count / jobs_count >= 0.005

    return False

# Define the regular expression pattern for tokenization
title_pattern = r"\s*[-,()]+\s*|\s*\.\s+|\s*\.$"

# Create a RegexpTokenizer with the defined pattern
TITLE_TOKENIZER = RegexpTokenizer(title_pattern, True)

def parse_job_titles_by_phrases(job_title):
    positionTokenSplit = TITLE_TOKENIZER.tokenize(job_title)
    phrases = set()
    for token in positionTokenSplit:
        words = token.lower().split()
        words = list(filter(lambda x: x not in IGNORED_WORDS, words))
        for i in range(len(words)):
            for j in range(1, min(5, len(words) - i + 1)): # Consider phrases of length 1 to 4 words
                phrase = ' '.join(words[i:i+j])
                phrases.add(phrase)

    return phrases

# Define the regular expression pattern for tokenization
post_pattern = r"\s*[-/,();:&]+\s*|\s*[!?.]$"

# Create a RegexpTokenizer with the defined pattern
POST_TOKENIZER = RegexpTokenizer(post_pattern, True)

def parse_job_posts_by_skills(job_post):
    parsed_skills = []
    sentences = [ # tokenize by sentences and also \n then replace out &nbsp; and trim
        sentence.replace(u'\xa0', u' ').strip()
        for s in nltk.sent_tokenize(job_post)
        for sentence in s.lower().split('\n')
    ]
    for sentence in sentences:
        if len(sentence):
            words = [w for ws in POST_TOKENIZER.tokenize(sentence) for w in ws.split()]

            line_skills = []
            for word in words:
                if word in SKILLS:
                    line_skills.append(word)

            if line_skills:
                parsed_skills.append(line_skills)

    return parsed_skills

def count_words(job_post, job_posts_word_count):
    sentences = [ # tokenize by sentences and also \n then replace out &nbsp; and trim
        sentence.replace(u'\xa0', u' ').strip()
        for s in nltk.sent_tokenize(job_post)
        for sentence in s.lower().split('\n')
    ]
    for sentence in sentences:
        if len(sentence):
            words = [w for ws in POST_TOKENIZER.tokenize(sentence) for w in ws.split()]

            if SKILLS and not any([word in SKILLS for word in words]):
                continue
                
            else:
                for word in words:
                    if word not in STOP_WORDS:
                        if word in job_posts_word_count:
                            job_posts_word_count[word] += 1
                        else:
                            job_posts_word_count[word] = 1

def retrieve_mongo_jobs():
    connection = MongoClient(CONNECTION_STRING)
    jobsDB = connection["JobSearchDB"]
    jobsCollection = jobsDB["jobs"]
    return jobsCollection, jobsCollection.find().sort('datePosted', -1)

def parse_jobs():
    jobsColl, jobs = retrieve_mongo_jobs()
    ml_phrases = list(set(GOOD_PHRASES + BAD_PHRASES + PERTINENT_PHRASES))
    full_data = []
    data = []
    num_conlumns = len(ml_phrases) + 1
    # Base case of failing zero phrases matched
    for _ in range(20):# average of 4 chosen for training set
        data.append([False] * num_conlumns)
    full_data = []
    ml_phrase_index = {}
    for i in range(1, num_conlumns):
        ml_phrase_index[ml_phrases[i-1]] = i

    phrase_counter = {}
    job_posts_word_count = {}
    likes = [0,0,0] # dislike, not rated, liked
    for job in jobs:
        if "liked" in job:
            liked = job["liked"]
            if liked == True:
                likes[2] += 1
                liked = 1
            else:
                likes[0] += 1
                liked = -1
        else:
            likes[1] += 1
            liked = 0

        phrases = parse_job_titles_by_phrases(job["position"])
        row = [False] * num_conlumns
        for phrase in phrases:
            if phrase in phrase_counter:
                phrase_counter[phrase][1+liked] += 1
            else:
                like_counts = [0,0,0]
                like_counts[1+liked] += 1
                phrase_counter[phrase] = like_counts
            
            if phrase in ml_phrases:
                    row[ml_phrase_index[phrase]] = True

        if liked:
            if liked == 1:
                row[0] = True
            data.append(row)
        else:
            full_data.append(row + [job['_id']])
            
        # TODO: logic to rank skills and title phrases
        post_skills = parse_job_posts_by_skills(job["fullJobPost"])
        # use for initially finding skill words from job posts
        count_words(job["fullJobPost"], job_posts_word_count)
        
    jobs_count = sum(likes)
    phrase_counter = list(filter(lambda x: x[1][0] + x[1][2] > 0 and occurrence_cutoff(x[0], sum(x[1]), jobs_count), phrase_counter.items()))
    phrase_counter.sort(key=lambda x: sum(x[1]))
    # print(list(map(lambda x: x[0], phrase_counter)))
    pprint(phrase_counter[-50:])
    
    for skill in SKILLS:
        if skill in job_posts_word_count:
            del job_posts_word_count[skill]

    job_posts_word_count = list(job_posts_word_count.items())
    job_posts_word_count.sort(key=lambda x: x[1])
    print(len(job_posts_word_count))
    pprint(job_posts_word_count[-20:])

    dep_var = 'job_liked'
    df = pd.DataFrame(data, columns=[dep_var] + ml_phrases)

    # Split the data into training and validation sets
    splits = RandomSplitter(valid_pct=0.2, seed=42)(range_of(df))

    # Define a tabular data loader
    to = TabularPandas(df, procs=[Categorify, FillMissing], cat_names = ml_phrases, y_names=dep_var, splits=splits)

    # Define your model
    dls = to.dataloaders(bs=64)

    # Define and train the model
    learn = tabular_learner(dls, metrics=accuracy)
    learn.fine_tune(52)

    # Evaluate your model on the validation set
    learn.show_results()

    fdf = pd.DataFrame(full_data, columns=[dep_var] + ml_phrases + ['_id'])

    dl = learn.dls.test_dl(fdf)
    predictions = learn.get_preds(dl=dl)

    updates = []

    for i, row in fdf.iterrows():
        updates.append(UpdateOne(
            {'_id': row['_id']},
            {'$set': {'titleRanking': float(predictions[0][i][1])}},
        ))

    jobsColl.bulk_write(updates)

if __name__ == "__main__":
    parse_jobs()