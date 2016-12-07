# Notes:
# Lose 2.4% of questions because the entids are not in feebase-ents.txt


from pyspark import SparkContext, SparkConf
import operator
import re, string
from nltk import ngrams
from collections import defaultdict
import operator as op
import math
import numpy as np
import sys
from googleapiclient import discovery
import httplib2
from oauth2client.client import GoogleCredentials
import cPickle
import os
from nltk.stem.porter import *

########################################################################
### List and Dict Helpers
########################################################################

# +, {'a':1,'b':2}, {'a':1,'c':3} -> {'a':2,,'b':2,c':3}
def outer_join_dicts(f,d1,d2) :
	d3 = {}
	for k in set.union(set(d1.keys()),set(d2.keys())) :
		d3[k] = f( d1[k] if k in d1 else 0 , d2[k] if k in d2 else 0 )
	return d3

# x^2, {'a':2} -> {'a':4}
def map_values(f,d) :
	return dict(map(lambda (k,v): (k, f(v)), d.iteritems()))

# [('a',1),('a',2),('b',1),('b',1)] -> {'a':[1,2],'b':[1,1]}
def group_pairs_by_first(pairs) :
	d = defaultdict(list)
	for pair in pairs :
		d[pair[0]].append(pair[1])
	return d

# [[[1,2],[3,4]],[[5,6]]] -> [[1,2],[3,4],[5,6]]
def flatten_one_layer(l) :
    return [item for sublist in l for item in sublist]

# {'a':1,'b':2,'c':2} -> {'a':0.2,'b':0.4,'c':0.4}
def normalize_dict(d) :
	sm = reduce(op.add,d.values())
	d = map_values(lambda x: x/sm, d)
	return d

# {'dog':0.2,'cat':0.1} + {'dog':0.3,'giraffe':0.1} -> {'dog':0.6}
def dict_dotproduct(bag1,bag2) :
	score = 0.0
	for key in set.intersection(set(bag1.keys()),set(bag2.keys())) :
		score += bag1[key]*bag2[key]
	return score


########################################################################
### String Helpers
########################################################################

# 'I go  there' -> 'I go there'
def remove_double_spaces(s) :
	return ' '.join(filter(lambda x: x!='', s.split(' ')))

# 'I do have a dog' + 'do' + 'did' -> 'I did have a dog'
def replace_complete_word(s,word,replacement) :
	return (' '+s+' ').replace(' '+word+' ',' '+replacement+' ')[1:-1]


########################################################################
### Vector Helpers
########################################################################

# ['dog','cat'],{'dog':2,'cat':4},6 -> [0,0,1,0,1,0]
def sparse_vector_to_dense_vector(l,key2id,length) :
	vec = np.zeros(length)
	for key in l :
		if key not in key2id :
			continue
		vec[key2id[key]] = 1
	return vec

# {'dog':0.2,'cat':0.3},{'dog':2,'cat':4},6 -> [0,0,0.2,0,0.3,0]
def bag_to_dense_vector(bag,key2id,length) :
	vec = np.zeros(length)
	for key in bag :
		if key not in key2id :
			continue
		vec[key2id[key]] = bag[key]
	return vec

def write_vector_as_csv(filename,vector) :
	with open(filename,'w') as f :
		for pt in vector :
			f.write(str(pt)+'\n')

def write_2d_matrix_as_csv(filename, matrix) :
	with open(filename,'w') as f :
		for row in matrix :
			s = ''
			for val in row :
				s += str(val)+','
			f.write(s[:-1]+'\n')

# {'a':0.2,'b':0.3}, 0.25 -> {'b':0.3}
def threshold_bag(d,thresh) :
	d2 = {}
	for key in d :
		if d[key] > thresh :
			d2[key] = d[key]
	return d2

# {'a':1,'b':2} + ['a','c'] -> {'a':2,'b':2,'c':1}
def add_list_to_count_dict(d,l) :
	for x in l :
		if x not in d :
			d[x] = 0
		d[x] += 1

# ['a','b','c'] + 1 -> {'a':1,'b':2,'c':3}
def list_to_id_map(l,start_id) :
	d = {}
	i = start_id
	for x in l :
		d[x] = i
		i += 1
	return d


########################################################################
### Load Word Resources (ie. word frequencies and filter words)
########################################################################

# global_word_log_cts:  {'the':24.12,'cat':2.33,...}
# filter_words:  set('the','a',...)
def load_word_resources() :
	with open('global_word_cts.txt','r') as f :
		lines = f.read().split('\n')
	global_word_log_cts = defaultdict(float)
	for line in lines :
		parts = line.split()
		global_word_log_cts[parts[0]] = math.log(int(parts[1]))

	with open('filter-words-100.txt','r') as f :
		filter_words = set(f.read().split('\n'))

	stemmer = PorterStemmer()

	return (global_word_log_cts, filter_words, stemmer)


########################################################################
### Store entity id to name mapping
########################################################################

# 'David	freebase-entity	<http://rdf.freebase.com/ns/m/067sbmt>	.' -> ('/067sbmt','david')
def make_uid_name_pair(line) :
	name = line[:line.find('\t')]
	# get rid of (...) in line because not part of entity name
	if '(' in name :
		name = name[:name.find('(')]
	name = re.sub('[^a-zA-Z0-9\' \-]+', '', name).lower().strip()
	uid = line[line.rfind('/'):line.rfind('>')]
	# if uid was parsed incorrectly then throw it out
	if ' ' in uid or len(uid)>10 :
		uid = ''
		name = ''
	return (uid,name)

# Write id->name mappings from raw freebase ents file
def process_entity_file(sc) :
	uid_name_pairs = sc.textFile("freebase-ents.txt").map(make_uid_name_pair) # [('1':'a'),('2','b'),...]
	unique_uid_name_pairs = uid_name_pairs.reduceByKey(lambda a,b: a) # get rid of duplicate ids
	unique_uid_name_pairs.coalesce(1).saveAsSequenceFile("entid2name") # condense to single file


###########################################################################
### Make list of all entity ids that we need to exact match
###########################################################################

# line:  www.freebase.com/m/03_7vl	www.freebase.com/people/person/profession	www.freebase.com/m/09jwl
# returns ['/067sbmt','/027vb3h',...]
def get_all_ids(line) :
	parts = line.split('\t')
	uid1 = parts[0]
	uid1 = uid1[uid1.rfind('/'):]
	l = []
	l.append(uid1)
	# can be multiple objects in relationship description so get all of them
	for i in range(2,len(parts)) :
		uid2 = parts[i]
		uid2 = uid2[uid2.rfind('/'):]
		l.append(uid2)
	return l

# Get list of entity ids that appear in the rulebase
def process_entid_list(sc) :
	ent_ids = sc.textFile("freebase-rules.txt").map(get_all_ids).flatMap(lambda x: x) # ['/067sbmt','/027vb3h',...]
	ent_ids.distinct().coalesce(1).saveAsTextFile("present-entids2") # get distinct ids and condense to single file


###########################################################################
### Make reduced ent_id to name map that only has entities present in the ruleset
###########################################################################

# make mapping of entid->name but only for entities in ruleset
def process_entname_list(sc) :
	present_entids = sc.textFile('present-entids/part-00000') # ['/067sbmt','/027vb3h',...]
	entid2name = sc.sequenceFile('entid2name/part-00000') # {'/067sbmt':'david','/027vb3h':'john',...]
	present_id_map = present_entids.map(lambda x: (x,1)) # convert from list to pairs so can join
	entid2name.join(present_id_map).coalesce(1).saveAsTextFile("entid2name-important") # write id2name map

# load entid->name mapping
#( {'/0a2':'david','/g5h':'steven',...}, {'david':['/0a2'],'steven':['/g5h'],...}, set('david','steven',...) )
def load_ent2name() :
	with open('entid2name-important/part-00000','r') as f :
		lines = f.read().split('\n')
		# (u'/012fh', (u'afrikaner', 1)) -> ('/012fh','afrikaner')
		pair_list = [(line[3:line.find('\',')],remove_double_spaces(line[line.find(' (u')+4:-6])) for line in lines]
		entid2name_important = dict( pair_list )
		entname2id_important = defaultdict(list) # can be multiple ids per name
		# make reversed map
		for id in entid2name_important :
			entname2id_important[ entid2name_important[id] ].append(id)
		entname_set = set( [tuple(s.split()) for s in entid2name_important.values()] )
	return (entid2name_important, entname2id_important, entname_set)


###########################################################################
### Load rules into memory
###########################################################################

# line: www.freebase.com/m/03_7vl	www.freebase.com/people/person/profession	www.freebase.com/m/09jwl 
# return (/03_7vl,/people/person/profession)
def process_rule_line(line) :
	parts = line.split('\t')
	uid1 = parts[0]
	uid1 = uid1[uid1.rfind('/'):]
	reltype = parts[1]
	reltype = reltype.replace('www.freebase.com','')
	return (uid1,reltype)

# Extract all distinct (uid1,relationship_type) pairs and write to a single file
def process_rules(sc) :
	sc.textFile("freebase-rules.txt").map(process_rule_line).distinct().coalesce(1).saveAsTextFile("rules")

# rules: {'/a2df':['profession,born_here'],...}
def load_rules() :
	rules = defaultdict(list)
	with open('rules/part-00000','r') as f :
		lines = f.read().split('\n')
		for line in lines :
			# line: "('/a2df','profession')"
			parts = line.split(',')
			id = parts[0][3:-1]
			rel = parts[1][3:-2]
			rules[id].append(rel)
	return rules


###########################################################################
### Quickly find possible mispellings by method from http://emnlp2014.org/papers/pdf/EMNLP2014171.pdf
### 1. make a distinct letter -> prime number mapping
### 2. multiply primes for letters in word
### 3. find all entities with scores that are off by one or two prime factors (off by one or two letters)
### 4. run edit distance on this vastly reduced set of candidates to find if the incorrect letters are properly positioned
###########################################################################

# entname_set: ['star wars','star trek']
# returns ( {'a':2,'b':3,...}, {'star wars':1.232424e46,...}, [2,3,0.5,0.33,1.5,...] )
def make_mispelling_resources(entname_set) :

	# map letters to prime numbers
	primes_letters = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101]
	primes_numbers = [103,109,113,127,131,137,139,149,151,157]
	primes_all = primes_letters + primes_numbers + [163,167,173]
	primes_map = {' ':163,'-':167,'\'':173}
	for i in range(26) :
		primes_map[chr(ord('a')+i)] = primes_letters[i]
	for i in range(10) :
		primes_map[chr(ord('0')+i)] = primes_numbers[i]

	# list of factors that entity letter score can be off by for one or two errors
	possible_spelling_ratios = set( flatten_one_layer([[1.0*x*y,1.0*x/y,1.0*y/x,1.0/x/y] for x in primes_all for y in primes_all])
				+ flatten_one_layer([[1.0*x,1.0/x] for x in primes_all]) )

	# map of spelling score to entity
	ent_spell_scores = {}
	for ent in entname_set :
		num_list = [primes_map[c] for c in ' '.join(ent)]
		if len(num_list)==0 or len(num_list)>40 :
			continue
		ent_spell_scores[float(reduce(op.mul,num_list))] = ent

	return (primes_map, ent_spell_scores, possible_spelling_ratios)

# source: http://stackoverflow.com/questions/2460177/edit-distance-in-python
def edit_distance(s1, s2):
	if len(s1) > len(s2):
		s1, s2 = s2, s1
	distances = range(len(s1) + 1)
	for i2, c2 in enumerate(s2):
		distances_ = [i2+1]
		for i1, c1 in enumerate(s1):
			if c1 == c2:
				distances_.append(distances[i1])
			else:
				distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
		distances = distances_
	return distances[-1]

# return list of entities off by 1 or 2 letters from ent
def find_mispellings(ent, primes_map, ent_spell_scores, possible_spelling_ratios) :

	# check each of ~1000 values that this spelling can be off by
	# add to possibilities any value that is present in ent_spell_scores so corresponds to a known entity
	find_val = reduce(op.mul,[primes_map[c] for c in ' '.join(ent)])
	possibilities = []
	for ratio in possible_spelling_ratios :
		if find_val*ratio in ent_spell_scores :
			possibilities.append(ent_spell_scores[long(find_val*ratio)])

	# use expensive edit distance method on reduced list to account for letter order
	found_ents = []
	for poss in possibilities :
		if edit_distance(' '.join(poss),' '.join(ent))<=2 :
			found_ents.append((poss,ent))
	return found_ents


###########################################################################
### Dependency Parse
###########################################################################

# the dog runs
# pos: {'the':'det','dog':'nn','runs':'vb'}
# deps: [[nsubj,dog,1,runs,2],[det,dog,1,the,0]]
# root: runs
class Parse:
	def __init__(self,pos,deps,root) :
		self.pos = pos
		self.deps = deps
		self.root = root
	
	def __str__(self) :
		s='['
		s+=self.pos.__str__()+',\n'
		s+=self.deps.__str__()+',\n'
		s+=self.root.__str__()+'\n'
		return s

	def __repr__(self) :
		return self.__str__()

# ie. [nsubj,dog,1,runs,2]
class Dep:
	def __init__(self,rel,w1,w1id,w2,w2id) :
		self.rel = rel
		self.w1 = w1
		self.w1id = w1id
		self.w2 = w2
		self.w2id = w2id

	def __str__(self) :
		return '[%s,%s,%i,%s,%i]' % (self.rel, self.w1, self.w1id, self.w2, self.w2id)
    
	def __repr__(self) :
		return self.__str__()

# Returns the encoding type that matches Python's native strings (source: stack overflow)
def get_native_encoding_type():
	if sys.maxunicode == 65535:
		return 'UTF16'
	else:
		return 'UTF32'

# Call analyze_text for Google NLP API then formats response to Parse object
def format_parse(extracted_info) :
	tokens = extracted_info.get('tokens', [])

	# extract word ids and part of speech tags
	words = {}
	pos = {}
	for i in range(len(tokens)) :
		token = tokens[i]
		word = token['text']['content']
		words[i] = word
		tag = token['partOfSpeech']['tag']
		pos[word] = tag.lower()

	# extract dependencies
	deps = []
	for i in range(len(tokens)) :
		token = tokens[i]
		dep = token['dependencyEdge']
		other_word = words[ dep['headTokenIndex'] ]
		other_word_index = dep['headTokenIndex']
		deps.append( Dep(dep['label'].lower(), words[i], i, other_word, other_word_index) )

	# find root
	for i in range(len(deps)) :
		if deps[i].w1id==deps[i].w2id :
			root = deps[i].w1

	return Parse(pos, deps, root)

# Call Google Natural Language syntax API, raises HTTPError is connection problem.
def parse_text(text):
  credentials = GoogleCredentials.get_application_default()
  scoped_credentials = credentials.create_scoped(['https://www.googleapis.com/auth/cloud-platform'])
  http = httplib2.Http()
  scoped_credentials.authorize(http)
  service = discovery.build(
      'language', 'v1beta1', http=http)
  body = {
      'document': {
          'type': 'PLAIN_TEXT',
          'content': text,
      },
      'features': {
          'extract_syntax': True,
          #'extract_entities': True,
      },
      'encodingType': get_native_encoding_type(),
  }
  request = service.documents().annotateText(body=body)
  extracted_info = request.execute()
  
  parse_info = format_parse(extracted_info)
  parse_string = cPickle.dumps(parse_info).replace('\n','\t')
  return parse_string


###########################################################################
### Training Flow Helpers
###########################################################################

# extract information from dataset line
def process_dataset_line(line,entid2name) :
	id1 = line[0].replace('www.freebase.com/m','')
	rel_type = line[1].replace('www.freebase.com','')
	id2 = line[2].replace('www.freebase.com/m','')
	# preprocess input text
	text = filter(lambda c: c in [' ','-','\''] or (c>='a' and c<='z') or (c>='0' and c<='9'), line[3].lower())
	text = re.sub(r"'(?=[a-z])", r" '", text)
	if id1 not in entid2name :
		ent1 = None
	else :
		ent1 = entid2name[id1]
	return (text,id1,ent1,rel_type)

# generate list of all possible entities
def generate_grams(text, filter_words) :
	words = text.split()
	# get rid of unigrams that are really common
	unigrams = filter(lambda tup: tup[0] not in filter_words, list(ngrams(words,1)))
	grams_list = unigrams
	for i in range(2,len(words)+1) :
		grams_list += list(ngrams(words,i))
	grams = set(grams_list)
	return grams

# make a list of possible mispelled entities from the text input
def generate_mispelled_ents(grams, exact_match_ents, global_word_log_cts, primes_map, ent_spell_scores, possible_spelling_ratios) :
	mispelled_ents = []
	# check all ngrams that we have not already exact matched
	for ent in grams-set([x[0] for x in exact_match_ents]) :
		# throw out candidate if very short length or only has very common words
		if len(' '.join(ent))<=4 or all([global_word_log_cts[w]>12 for w in ent]) :
			continue
		poss_ents = find_mispellings(ent, primes_map, ent_spell_scores, possible_spelling_ratios)
		mispelled_ents += poss_ents
	return mispelled_ents

# get the log probability of the phrase appearing randomly
def get_ent_score(ent_words, global_word_log_cts) :
	return reduce(op.add,[math.log(360000000)-global_word_log_cts[w] for w in ent_words])

# ['a','b','c','d'] -> {'a':0.25,'b':0.25,'c':0.25,'d':0.25}
def uniform_normalized_bag(l) :
	return dict( zip(l,[1.0/len(l) for x in l]) )

# get tf dotproduct score for all bags relative to the word weights
def get_rel_scores(word_weights, relationship_bags) :
	scores = []
	for rel in relationship_bags :
		scores.append( ( rel, dict_dotproduct(word_weights,relationship_bags[rel]) ) )
	return dict(scores)

# get all possible rules for entity
def get_present_rels(ent_words, entname2id, rules) :
	ids = entname2id[' '.join(ent_words)] # get ids for entity name
	rules_list = []
	for id in ids :
		rules_list += rules[id]
	return set(rules_list)

