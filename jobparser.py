from pymongo import MongoClient
from pprint import pprint
import nltk
# nltk.download('punkt') # uncomment on first run to get download
# nltk.download('stopwords') # uncomment on first run to get download
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

CONNECTION_STRING = "mongodb://<user>:<pass>@<host>/<DBName>"
STOP_WORDS = stopwords.words('english')
IGNORED_WORDS = ["senior", "sr", "/", "&", "mission", "it", "and", "of", "mn", "in", "a"] + STOP_WORDS
BAD_PHRASES = [
 'salesforce', 'salesforce developer', 'salesforce devops admin',
 'manager', 'engineering manager',
 'clearance required',
 'lead', 'lead software', 'lead software engineer',
 'analytics',
 'solution architect',
 'workflow design studio',
 'director',
]
GOOD_PHRASES = [
 'embedded',
 'go',
 'react',
 'web',
 'back end',
 'automation',
 'backend',
 'java',
 'hybrid',
 'python',
 'test',
 'remote',
 'iii',
 'reliability engineer', 'site reliability', 'site reliability engineer', 'site reliability engineering',
 'full stack',
 'developer',
 'full stack developer',
 'software', 'staff software', 'software developer', 'software development',
 'engineer', 'engineer iii', 'engineer ii', 'engineering',
 'devops engineer', 'network engineer', 'automation engineer', 'development engineer',
 'software engineer', 'software engineering', 'software engineer ii', 'software engineer iii',
 'embedded software engineer', 'software development engineer',
]
NEUTRAL_SKILLS = [
 'sql',
 'azure',
 'aws',
 'web',
 'etl', 
 'scripting',
 'databases',
 'devops',
 'database',
 'saas', 
 'crm', 
 'css', 
 '.net', 
 'algorithms', 
 'c#', 
 'html', 
 'sre', 
 's3', 
 'qa', 
 'ip', 
 'tableau', 
 'r', 
 'matlab', 
 'elt', 
 'xml', 
 'jira', 
 'powershell', 
]
BAD_SKILLS = [
 'salesforce',
 'servicenow',
 'ad', 
 'sap', 
]
GOOD_SKILLS = [
 'javascript',
 'java',
 'python',
 'c++', 
 'react',
 'linux',
 'docker',
 'kubernetes',
 'orchestration', 
 'go',
 'scale',
 'jenkins', 
 'nosql', 
 'scalability', 
 'angular', 
 'backend', 
 'embedded', 
 'git', 
 'container', 
 'terraform', 
 'typescript', 
 'rest', 
 'c', 
 'spring', 
 'node.js', 
 'android', 
 'scala', 
 'mysql', 
 'apache', 
 'ruby', 
 'frontend', 
 'oracle', 
 'github', 
 'bash', 
 'microservices', 
 'shell', 
 'mongodb', 
 'postgres', 
 'node', 
 'postgresql', 
 'fpga', 
 'unix', 
 'rdbms',
 'selenium', 
 'containerization',
 'gitlab', 
 'dynamodb', 
 'firmware', 
 'kotlin', 
 'json', 
 'restful', 
 'datadog', 
 'lambda', 
 'kafka', 
 'ansible', 
]
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
    return jobsCollection.find()

def parse_jobs():
    jobs = retrieve_mongo_jobs()

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
        for phrase in phrases:
            if phrase in phrase_counter:
                phrase_counter[phrase][1+liked] += 1
            else:
                like_counts = [0,0,0]
                like_counts[1+liked] += 1
                phrase_counter[phrase] = like_counts
            
        parsed_skill = parse_job_posts_by_skills(job["fullJobPost"])
        # use for inirially finding skill words from job posts
        count_words(job["fullJobPost"], job_posts_word_count)
        
    jobs_count = sum(likes)
    phrase_counter = list(filter(lambda x: occurrence_cutoff(x[0], sum(x[1]), jobs_count), phrase_counter.items()))
    phrase_counter.sort(key=lambda x: sum(x[1]))
    pprint(phrase_counter[-50:])
    
    for skill in SKILLS:
        if skill in job_posts_word_count:
            del job_posts_word_count[skill]

    job_posts_word_count = list(job_posts_word_count.items())
    job_posts_word_count.sort(key=lambda x: x[1])
    print(len(job_posts_word_count))
    pprint(job_posts_word_count[-20:])

if __name__ == "__main__":
    parse_jobs()