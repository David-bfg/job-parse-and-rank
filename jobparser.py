from pymongo import MongoClient
from pprint import pprint
import nltk
# nltk.download('punkt') # uncomment on first run to get download
# nltk.download('stopwords') # uncomment on first run to get download
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

CONNECTION_STRING = "mongodb://<user>:<pass>@<host>/<DBName>"
IGNORED_WORDS = ["senior", "sr", "/", "&", "mission", "it", "and", "of", "mn", "in", "a"]
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

def occurrence_cutoff(phrase, phrase_count, jobs_count):
    if phrase_count == 1:
        return False

    match len(phrase.split()):
        case 1:
            return phrase_count / jobs_count >= 0.01
        case 2:
            return phrase_count / jobs_count >= 0.01
        case 3:
            return phrase_count / jobs_count >= 0.005
        case 4:
            return phrase_count / jobs_count >= 0.005

    return False


def mongo_connect():
    connection = MongoClient(CONNECTION_STRING)
    jobsDB = connection["JobSearchDB"]
    jobsCollection = jobsDB["jobs"]
    jobs = jobsCollection.find()

    # Define the regular expression pattern for tokenization
    pattern = r"\s*[-,()]+\s*|\s*\.\s+|\s*\.$"

    # Create a RegexpTokenizer with the defined pattern
    name_tokenizer = RegexpTokenizer(pattern, True)

    # Tokenize the input text
    phrase_counter = {}
    jobs_count = 0
    job_posts = []
    for job in jobs:
        jobs_count += 1
        positionTokenSplit = name_tokenizer.tokenize(job["position"])
        job_posts.append(job["fullJobPost"])
        for token in positionTokenSplit:
            words = token.lower().split()
            words = list(filter(lambda x: x not in IGNORED_WORDS, words))
            for i in range(len(words)):
                for j in range(1, min(5, len(words) - i + 1)): # Consider phrases of length 1 to 4 words
                    phrase = ' '.join(words[i:i+j])
                    if phrase in phrase_counter:
                        phrase_counter[phrase] += 1
                    else:
                        phrase_counter[phrase] = 1
                    
    phrase_counter = list(filter(lambda x: occurrence_cutoff(x[0], x[1], jobs_count), phrase_counter.items()))
    phrase_counter.sort(key=lambda x: x[1])
    pprint(phrase_counter[-50:])
    
    # Define the regular expression pattern for tokenization
    pattern = r"\s*[-/,();:&]+\s*|\s*[!?.]$"

    # Create a RegexpTokenizer with the defined pattern
    post_tokenizer = RegexpTokenizer(pattern, True)

    job_post_word_count = {}
    stop_words = stopwords.words('english')
    skills = set(GOOD_SKILLS + NEUTRAL_SKILLS + BAD_SKILLS)
    for post in job_posts:
        sentences = [
            sentence.replace(u'\xa0', u' ').strip()
            for s in nltk.sent_tokenize(post)
            for sentence in s.lower().split('\n')
        ]
        for sentence in sentences:
            if len(sentence):
                words = [w for ws in post_tokenizer.tokenize(sentence) for w in ws.split()]
                if not any([word in skills for word in words]):
                    continue
                
                for word in words:
                    if word not in stop_words:
                        if word in job_post_word_count:
                            job_post_word_count[word] += 1
                        else:
                            job_post_word_count[word] = 1
    
    for skill in skills:
        if skill in job_post_word_count:
            del job_post_word_count[skill]

    job_post_word_count = list(job_post_word_count.items())
    job_post_word_count.sort(key=lambda x: x[1])
    print(len(job_post_word_count))
    pprint(job_post_word_count[-600:])

if __name__ == "__main__":
    mongo_connect()