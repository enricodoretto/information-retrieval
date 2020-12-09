# **Information Retrieval Course Project**

This project was built for the Information Retrieval 2020/2021 Course.

It consists in a Python implementation of a Boolean Information Retrieval System that can answer AND, OR, NOT queries as well as phrase queries using a positional index and wildcard queries using permuterm index.

##Python dependencies
- nltk
- functools
- re
- csv
- time
- pickle
- sys
- gc

##Initialization
- Start the program
- Press "i" to build the index the first time
- The index will be automatically saved

#####Operation performed
- Normalization removing punctuation and putting to lower case
- Tokenization splitting the words by space
- Stop words removal 
- Porter Stemmer

##Perform queries
- Start the program after the index is built
- Press "q" to perform a query
- Press "i" to automatically load the index
- Once the index is loaded press "q" and write the query

#####And, Or, Not queries
- Write the query in the form _term operator term_ 
- The operators can be _and_, _or_, _and not_
- If you want all documents where only a term is not present the query will be _not term_
- You can have more than a single operator in the query, like _term operator term operator term .._ 

#####Phrase queries
- Simply put two or more terms in order separated by a space, like _term term term_

#####Wildcard queries
- Put an '#' in the part of the term you don't know, like _te#m_
- You can also perform multiple wildcards putting more '#', like _#e#m_

#####All queries combined
- You can perform all kind of query together like _term1 and term2 term3 and not term4_ where single terms can be wilcards
- Priority will always be given to phrase queries 