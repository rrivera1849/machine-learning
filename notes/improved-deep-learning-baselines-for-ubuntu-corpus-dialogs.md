
# Improved Deep Learning Baselines for Ubuntu Corpus Dialog

Ubuntu Dialog Corpus used as data, the proposed task is that of response 
selection. They explore three approaches: pointwise, pairwise and listwise 
ranking.

Pointwise ranking simply takes a context and a response and outputs a probability. 
It is of the form g(context, response) = probability. 

TF-IDF: The idea here is that the correct response shares more words with the 
context than incorrect responses. This corresponds to taking a dotproduct between 
the vectorized context and response vectors.

Neural Network context and reponse embeddings: 
Given some neural network function, we can compute a similarity metric as follows:

    c = f(context)
    r = f(response)
    g(context, response) = \sigma(c^T M r + b)

This can be thought of as predicting the response from the context as such:

    r' = c^T M

The dot product then measures the similarity between the predicted response and the 
correct reponse.

Three different neural networks are used to model the function: CNN, LSTM, Bi-LSTM.
Sharing parameters for both the context and response embeddings worked best. 

