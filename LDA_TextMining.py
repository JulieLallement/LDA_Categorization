#!/usr/bin/env python
# coding: utf-8

# # Installations

# In[1]:


pip install gensim


# In[2]:


pip install pyLDAvis


# In[3]:


pip install --upgrade pyLDAvis


# In[4]:


pip install nltk


# In[5]:


import pandas as pd
import nltk
import gensim


# In[10]:


df = pd.read_csv('df_lemme_V2.csv')


# In[11]:


df.head()


# In[17]:


df = df.dropna()


# # Nettoyage du corpus

# In[18]:


corpus = df['text_clean'].to_list()
print(corpus[0])


# In[20]:


corpus = [doc.lower() for doc in corpus] #passer en minuscule


# In[21]:


import string
ponctuations = list(string.punctuation)
print(ponctuations)


# In[22]:


#retrait des ponctuations
corpus = ["".join([char for char in list(doc) if not (char in ponctuations)]) for doc in corpus]
print(corpus[0])


# In[23]:


#tokenization 
nltk.download('punkt')
from nltk.tokenize import word_tokenize
corpus_tk = [word_tokenize(doc) for doc in corpus]


# In[66]:


#lemmatisation
nltk.download('omw-1.4')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lem = WordNetLemmatizer()
corpus_lm = [[lem.lemmatize(mot) for mot in doc] for doc in corpus_tk]
print(corpus_lm[6])


# In[26]:


#charger les stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords
mots_vides = stopwords.words('french')
print(mots_vides)


# In[45]:


#compléter la liste de stopwords
mots_vides.extend(['bonjour', 'tout', 'merci', 'faire', 'bien','monsieur', 'remercier', 'Monsieur', 'madame', 'Madame', 'cg', 'cerner', 'sincèrement', 'mondial', 'gestionnaire', 'prestation', 'centre', 'gestion', 'mondial', 'ced', 'alexandra', 'dune', 'suite', 'page', 'dossier', 'contrat', 'décembre', 'janvier', 'mai', 'avril', 'février', 'fax', 'mars', "juin", 'juillet', 'aout', 'septembre','novembre', 'siège', 'confidentiels', 'confidentiel', 'mile', 'logistique'])


# In[46]:


corpus_sw = [[mot for mot in doc if not mot in mots_vides] for doc in corpus_lm]


# In[47]:


#retirer les tokens de moins de 3 lettres
corpus_sw = [[mot for mot in doc if len(mot) >= 3] for doc in corpus_sw]


# # Topic modeling avec Gensim

# In[48]:


from gensim.corpora import Dictionary
dico = Dictionary(corpus_sw)
print(dico)
# 7706 tokens uniques


# In[49]:


print(dico.token2id)


# In[50]:


#représentation bag of words
corpus_bow = (dico.doc2bow(doc) for doc in corpus_sw)
corpus_bow_list = list(corpus_bow)
print(sorted(corpus_sw[1]))
print(corpus_bow_list[1])


# In[97]:


#LDA
from gensim.models import LdaModel
lda = LdaModel(corpus=corpus_bow_list,
               id2word=dico,
               num_topics=9, 
               chunksize = 200,
               iterations = 200,
               passes=150,
               distributed = False,
               update_every = 100,
               eta = 'auto',
               alpha = 'auto')


# In[98]:


#affichage des topics
lda.print_topics()


# In[99]:


#détail pour le premier topic - numéro 0
lda.show_topic(0)


# In[100]:


#topic où le terme "contrat" joue un rôle
lda.get_term_topics("paiement", minimum_probability =0)


# In[101]:


#description des documents dans l'espace des topics
doc_topics = lda.get_document_topics(corpus_bow_list)
print(doc_topics)


# In[102]:


#transformation en matrice "sparse"
from gensim.matutils import corpus2csc
mat_sparse = corpus2csc(doc_topics)
print(mat_sparse)
#montre topic avec valeur non nulle, et le numéro du document


# In[103]:


#et en matrice "normale"
mat_dt = mat_sparse.T.toarray()
print(mat_dt)


# In[104]:


print(mat_dt.shape)


# In[105]:


#df 
dfTopic = pd.DataFrame(mat_dt, columns = ["T"+str(i) for i in range(mat_dt.shape[1])])
#verif
print(dfTopic.head())


# # Visualisation

# In[106]:


for (topic, words) in lda.print_topics():
    print("***********")
    print("* topic", topic+1, "*")
    print("***********")
    print(topic+1, ":", words)
    print()


# In[107]:


import pyLDAvis.gensim
import warnings

pyLDAvis.enable_notebook()
warnings.filterwarnings("ignore", category=DeprecationWarning)

pyLDAvis.gensim.prepare(lda, corpus_bow_list, dico, sort_topics=False, n_jobs=3)


# In[108]:


cluster_indices = [max(lda.get_document_topics(doc), key=lambda x: x[1])[0] for doc in corpus_bow_list]


# In[109]:


cluster_documents = {}
for i, cluster in enumerate(cluster_indices):
    if cluster not in cluster_documents:
        cluster_documents[cluster] = []
    cluster_documents[cluster].append(i)


# In[110]:


top_documents_per_cluster = {}
for cluster, documents in cluster_documents.items():
    top_documents_per_cluster[cluster] = documents[:5]


# In[119]:


import matplotlib.pyplot as plt

topic_probabilities = dfTopic.mean()

# Création de l'histogramme
plt.bar(range(len(topic_probabilities)), topic_probabilities)
plt.xlabel('Numéro du Topic')
plt.ylabel('Probabilité d\'apparition')
plt.title('Probabilité d\'apparition des Topics dans le Corpus')
plt.xticks(range(len(topic_probabilities)), ["T"+str(i) for i in range(len(topic_probabilities))])
plt.show()


# In[129]:


# Créer un dictionnaire pour stocker les textes par topic
topics_texts = {i: [] for i in range(lda.num_topics)}

# Parcourir les documents et leurs distributions de sujets
for doc_id, doc_topic in enumerate(doc_topics):
    # Récupérer les topics les plus pertinents pour le document
    top_topics = sorted(doc_topic, key=lambda x: x[1], reverse=True)[:5]
    # Récupérer l'ID du document dans le corpus initial
    original_doc_id = df.index[doc_id]
    # Récupérer le texte du document
    text = df.loc[original_doc_id, "text_clean"]
    # Ajouter le texte du document à chaque topic correspondant
    for topic, _ in top_topics:
        topics_texts[topic].append(text)

# Afficher les 5 textes de chaque topic
for topic, texts in topics_texts.items():
    print("Topic", topic, ":")
    for i, text in enumerate(texts[:10]):
        print("- Document", i+1, ":", text)
    print()


# In[ ]:




