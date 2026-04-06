import os
import re
import time
import cProfile
import pstats
from collections import Counter
import gutenbergpy.textget
from joblib import Parallel, delayed
from concurrent.futures import ThreadPoolExecutor

os.makedirs('corpus', exist_ok=True)

books = {
    1260:  'jane_eyre',
    20765: 'heidi',
    64317: 'great_gatsby',
    45:    'anne_of_green_gables',
    174:   'dorian_gray',
    158:   'emma',
    84:    'frankenstein',
    113:   'secret_garden',
    236:   'jungle_book',
    55:    'wizard_of_oz',
}

def download_corpus():
    for book_id, name in books.items():
        filepath = 'corpus/' + name + '.txt'
        if os.path.exists(filepath):
            continue
        print('Downloading ' + name)
        raw = gutenbergpy.textget.get_text_by_id(book_id)
        clean = gutenbergpy.textget.strip_headers(raw)
        f = open(filepath, 'wb')
        f.write(clean)
        f.close()
        print('saved ' + filepath)


def tokenise(text):
    #lowercase the whole text and identify words
    text = text.lower()
    words = re.findall(r'\b[a-z]+\b', text)
    return words


def count_file(filepath):
    #count each word occurance 
    f = open(filepath, 'r', encoding='utf-8', errors='ignore')
    text = f.read()
    f.close()
    words = tokenise(text)
    counts = Counter(words)
    return counts


def merge(list_of_counters):
    #combines each books' wordcounts into a big dictionary
    total = Counter()
    for c in list_of_counters:
        total = total + c
    return total


def sequential(corpus_dir):
    #make baseline, sequential method.
    all_files = os.listdir(corpus_dir)
    files = []
    for filename in all_files:
        if filename.endswith('.txt'):
            files.append(os.path.join(corpus_dir, filename))

    #reading the files
    t0 = time.perf_counter()
    texts = {}
    for filepath in files:
        f = open(filepath, 'r', encoding='utf-8', errors='ignore')
        texts[filepath] = f.read()
        f.close()
    t1 = time.perf_counter()

    #tokenise each text
    tokenised = {}
    for filepath, text in texts.items():
        tokenised[filepath] = tokenise(text)
    t2 = time.perf_counter()

    #count words in each file
    local_counts = []
    for filepath, words in tokenised.items():
        local_counts.append(Counter(words))
    t3 = time.perf_counter()

    #merge all files counts together
    result = merge(local_counts)
    t4 = time.perf_counter()

    print('Reading:    ' + str(round(t1 - t0, 3)) + 's')
    print('Tokenising: ' + str(round(t2 - t1, 3)) + 's')
    print('Counting:   ' + str(round(t3 - t2, 3)) + 's')
    print('Merging:    ' + str(round(t4 - t3, 3)) + 's')
    print('Total:      ' + str(round(t4 - t0, 3)) + 's')

    return result #this gives us our baseline (sequential)


def parallel_joblib(corpus_dir, n_jobs=-1):
    all_files = os.listdir(corpus_dir)
    files = []
    for filename in all_files:
        if filename.endswith('.txt'):
            files.append(os.path.join(corpus_dir, filename))

    t0 = time.perf_counter()
    local_counts = Parallel(n_jobs=n_jobs)(delayed(count_file)(f) for f in files)
    t1 = time.perf_counter()

    result = merge(local_counts)
    t2 = time.perf_counter()

    print('Parallel counting: ' + str(round(t1 - t0, 3)) + 's')
    print('Merging:           ' + str(round(t2 - t1, 3)) + 's')
    print('Total:             ' + str(round(t2 - t0, 3)) + 's')

    return result


def parallel_threads(corpus_dir, n_workers=4):
    all_files = os.listdir(corpus_dir)
    files = []
    for filename in all_files:
        if filename.endswith('.txt'):
            files.append(os.path.join(corpus_dir, filename))

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        local_counts = list(executor.map(count_file, files))
    t1 = time.perf_counter()

    result = merge(local_counts)
    t2 = time.perf_counter()

    print('Thread counting: ' + str(round(t1 - t0, 3)) + 's')
    print('Merging:         ' + str(round(t2 - t1, 3)) + 's')
    print('Total:           ' + str(round(t2 - t0, 3)) + 's')

    return result


if __name__ == '__main__':
    download_corpus()

    print('\n--- Sequential ---')
    result = sequential('corpus')
    print(result.most_common(10))

    print('\n--- cProfile ---')
    cProfile.run('sequential("corpus")', 'profile_output')
    stats = pstats.Stats('profile_output')
    stats.sort_stats('tottime')
    stats.print_stats('prof-skills-lab-4')

    print('\n--- Parallel (joblib) ---')
    result_parallel = parallel_joblib('corpus')
    print(result_parallel.most_common(10))
    print('Results match?:', result == result_parallel)

    print('\n--- Parallel (threads) ---')
    result_threads = parallel_threads('corpus')
    print(result_threads.most_common(10))
    print('Results match?:', result == result_threads)