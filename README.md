# RNN_RecurrentNeuralNETWORKS
[RNN_TextCategorization_PN.py] In this code is based on python which is purpose perform text classification,which is automatically predicting classification labels (such as postive or negative in sentiment analysis) based on text content.
In this code the several Python libs is used (Pytorch-for neural networkds training \ pandas-The DATA analysis lib for reading and processing CSV files \ NumPy_ the scientific computing lib for processing and transforming Data \NLTK_word segmentation module in the natural lagnuage processing lib NLTK
This RNN features：（1）DATA PROCESSING: loading data--use the pandas lib to read text data and labels from CSV files (this required the text with the comments and each comments is correspond to the 1 and 0 represent the postive and negative sentiment as this code is used to perform sentiment analysis similar to the two classification problem.)----create a vocabulary : tokenize the text(word_tokenize) and create a vocabulary that maps each word to a unique index.----encoding text: converting text into a sequnece of numbers so it can be processed by a neural network.  (2) Pythorch datasets and data loaders: Define a custom PYtorch DATASET to store text data and corresponding labels Use dataloader to batch load data, which is necessary for training large neural networks.  (3) define a simple RNN model using Pytorch's nn.module. the model includes an embedding layer (nn.embedding ) that converts word indices into dence vector representations AN RNN layer (nn.RNN) that processes sequence data and captures the temporal dynamics in the sequnece Finally , there is a fully connected (全连接层 nn.Linear),which converts the output of the RNN layer into the final classificaiton prediction  (4)the process for the training 
