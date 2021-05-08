# Recommender System

For different use cases, there is a need for large scale (cluster computing) 
productionalized models, but it is not always the case. The large and small
scale models can be generally split by whether the model requires more than
16 gb of memory to run, and if the computation speed of one computer is 
'fast enough'.

Small scale solution will utilize sparse matrices to reduce memory 
overhead (as most user-item matrices will have many 0's) 