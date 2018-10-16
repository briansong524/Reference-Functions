### Splitting text into tuple (split_tuple) ###

def split_cat(text): # this one is to reduce the categoriy_name into three subcategories
    try: return text.split("/")
    except: return ("No Label", "No Label", "No Label")
	'''how its used
	train_raw['cat1'],train_raw['cat2'],train_raw['cat3'] = \
	zip(*train_raw['category_name'].apply(lambda x: split_cat(x))) # split the categories into three new columns
	train_raw.drop('category_name',axis = 1, inplace = True) # remove the column that isn't needed anymore
	'''



### Fill NaN values with a placeholder (fillnan) ###

def handle_missing_inplace(dataset):  # this one is to put placeholders in place of missing values (NaN)
	dataset['cat1'].fillna(value='No Label', inplace=True)




### Obtain vocab size by percentile (vocsizefind) ###

def obtain_reasonable_vocab_size(list_words, perc_words = .95):
    counter_ = Counter(list_words)
    counts = [i for _,i in counter_.most_common()]
    tot_words = len(list_words)
    print('total words (with repeats): ' + str(tot_words))
    tot_count = 0
    runs = 0
    while tot_count < round(perc_words*tot_words):
        tot_count += counts[runs]
        runs += 1
    print('reasonable vocab size: ' + str(runs))




### Building dictionary for a set of words (build_dict) ### 

def build_dictionary(words, n_words): # dictionary that maps words to indices. this function should be modular.
    '''
	'words' is tokenized list/pd.Series [['a','b','c'],['a','b','c']]
    'n_words' is the number of words that should be in the dictionary 
    ''' 

    """Process raw inputs into a dataset."""
    count = [['UNK', -1]] # word indexed as "unknown" if not one of the top n_words (popular/common) words (-1 is filler #)
    count.extend(Counter(words).most_common(n_words - 1)) # most_common returns the top (n_words-1) ['word',count]
    dictionary = dict()
    for word, _ in count: # the 'word, _' is writted because count is a list of list(2), so defining 'word' as the first term per
        dictionary[word] = len(dictionary) # {'word': some number incrementing by one. fyi, no repeats because from most_common}
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys())) # {ind. : 'word'} I guess for looking up if needed?
    return dictionary, reversed_dictionary



### Cleaning and tokenizing a list of string (cleantok) ###
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import re
def clean_and_tokenize(dataset_col): # input is a column of strings
    pattern = '[A-Za-z]+' # does this only keep words
    pattern2 = '[!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n]' # to rid of special characters
    list_of_lists = list()
    tokenizer = RegexpTokenizer(pattern)
    words_to_remove = ['description', 'yet','like','the','any','they'] # manually include words
    words_to_remove.extend(stopwords.words('english')) # the, and, a, etc
    words_to_remove = set(words_to_remove)
    
    for word in dataset_col:
        list_of_words = list()
        word = re.sub(pattern2, r'', word)
        tokenized = tokenizer.tokenize(word)
        tokenized_filtered = filter(lambda token: token not in words_to_remove, tokenized)
        for i in tokenized_filtered:
            if (len(i) > 2 ): #ignore words of length 2 or less
                list_of_words.append(i.lower()) # append all words to one list
        list_of_lists.append(list_of_words)
    list_as_series = pd.Series(list_of_lists)
    return list_as_series



### Convert list of words to a list of lists of indices (word2ind) ###
def convert_word_to_ind(dataset_col,dictionary): # input the pandas column of texts and dictionary. This should be modular
    # each input should be a string of cleaned words tokenized into a list (ex. ['this', 'is', 'an', 'item'])
    # dictionary should be the dictionary obtained from build_dictionary
    list_of_lists = []
    unk_count = 0 # total 'unknown' words counted
    for word_or_words in dataset_col: # words is the list of all words
        list_of_inds = []
        for word in word_or_words:
            if word in dictionary:
                index = np.int(dictionary[word]) # dictionary contains top words, so if in, it gets an index
            else:
                index = 0  #  or dictionary['UNK']? can figure out later
                unk_count += 1
            list_of_inds.append(index)
        list_of_lists.append(list_of_inds)

    # make list_of_lists into something that can be put into pd.DataFrame
    #list_as_series = pd.Series(list_of_lists)
    list_as_series = np.array(list_of_lists)
    return list_as_series, unk_count



### deciding pad lengths (findpadlen) ### 

group_name = np.hstack((train_raw.name_inds, test_raw.name_inds))
group_itemdesc = np.hstack((train_raw.item_desc_inds, test_raw.item_desc_inds))
a = [len(i) for i in group_name ]
print("max tokens in 'name': " + str(max(a)))

b = [len(i) for i in group_itemdesc]
plt.hist(b,20)
plt.show()
print("max tokens in 'item_desc': " + str(max(b)))

sort_b = sorted(b) #sorted length in increasing order
perc_data = .95
len_item_desc_potential = sort_b[round(perc_data*len(sort_b))]
print(len_item_desc_potential) # this represents (perc_data)% of item descriptions are under (len_item_desc_potential) words



### Convert a list of words to a padded sequence of indices (word2padind) ###

def convert_word_to_padded(dataset_col,dictionary,pad_length): 
	# input the pandas column of texts and dictionary. This should be modular
    # each input should be a string of cleaned words tokenized into a list (ex. ['this', 'is', 'an', 'item'])
    # dictionary should be the dictionary obtained from build_dictionary
    # use this function when you know how long you want your pad_length
    #   - otherwise, use convert_word_to_ind

    list_of_lists = []
    unk_count = 0 # total 'unknown' words counted
    for word_or_words in dataset_col: # words is the list of all words
        list_of_inds = []
        count_inds = 0
        for word in word_or_words:
            if word in dictionary:
                index = np.int(dictionary[word]) # dictionary contains top words, so if in, it gets an index
            else:
                index = 0  #  or dictionary['UNK']? can figure out later
                unk_count += 1
            count_inds +=1
            list_of_inds.append(index) 
        if count_inds >= pad_length:
            asdf = list_of_inds[(count_inds-pad_length):]
        else: 
            asdf = [0]*(pad_length-count_inds)
            asdf.extend(list_of_inds)
        temp = np.array(asdf)
        list_of_lists.append(temp)
    list_as_series = np.array(list_of_lists)
    return list_as_series, unk_count



### Generate batches to train a bag-of-words context model (batchbow) ###
def generate_batch(data, batch_size, num_skips): 
    # data should be [[3,7,9],[7,4,5,9],...] kinda format
    # num_skips configures number of context words to draw. skip_window defines size of window to draw context words from
    assert batch_size % num_skips == 0 # if batch_size was 10, and num_skips was 3, then [cat,cat,cat,sat,sat,sat,...] wont equal
    batch = np.ndarray(shape=(batch_size), dtype=np.int32) # initialize batch variable (input word go in here)
    context = np.ndarray(shape=(batch_size, 1), dtype=np.int32) # initialize context variable
    counter = 0
    rand_dat_ind = random.sample(range(0,len(data)-1),int(batch_size/num_skips))
    for i in data[rand_dat_ind]:
        while len(i) <= num_skips:
            rnd_again = random.randint(0,len(data)-1)
            i = data[rnd_again]
        target = random.randint(0,len(i)-1) 
        targets_to_avoid = [target] # avoid this index when selecting rando words
        for j in range(num_skips):
            while target in targets_to_avoid: # this is to choose an index that isnt the index of the batch word
                target = random.randint(0, len(i)-1) # target is a context word
            targets_to_avoid.append(target) # so next time, this loop won't select this context word again 
            batch[counter] = i[targets_to_avoid[0]]  # this is the input word (same word repeated i*num_skips+j times)
            context[counter, 0] = i[targets_to_avoid[j+1]]  # these are the context words to the batch word
            counter += 1
    return batch, context # batch is input, context is target variable(s)