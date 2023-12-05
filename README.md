# BERT fine-tuning (RNN text categorization)
The downstream task of this model is RNN text categorization, using bert to generate word vectors
Write language model for RNN and write RNN model for text categorization based on trained word vectors (references below)

Yang, Zichao, Diyi Yang, Chris Dyer, Xiaodong He, Alex Smola, and Eduard Hovy. "Hierarchical attention networks for document classification." In *Proceedings of the 2016 conference of the North American chapter of the association for computational linguistics: human language technologies*, pp. 1480-1489. 2016.

1. **network framework**：pytorch
2. **dataset**：Use the Yelp2013 dataset. Use test.json from the dataset as the test set and manually divide the training and validation sets from yelp_academic_dataset_review.json. Download link: https://github.com/rekiksab/Yelp/tree/master/yelp_challenge/yelp_phoenix_academic_dataset Use only STARS ratings and TEXT review content.
3. **Model building**: Use pytorch or the encapsulated module to write the model, torch.nn.Linear(), torch.nn.Relu(), etc. 4. **Model training**: Input the generated training set into the built model for forward loss computation and backward gradient propagation to train the model.
4. **Model training**: Input the generated training set into the built model for forward loss calculation and backward gradient propagation, so as to train the model, and it is also recommended to use the optimizer encapsulated by the network framework to complete the parameter updating process. During the training process, the loss of the model on the training set and validation set is recorded and visualized in plots.
5. **Tuning analysis**: The trained model is tested on the validation set, and **Top 1 Accuracy(ACC)** is used as the network performance index. Then, the dropout, normalization, learning rate decay, residual connection, network depth are tuned, and then retrained, tested, and analyzed for the effect on model performance.
6. **TEST PERFORMANCE**: Select the set of hyperparameters that performs best on the validation set, retrain the model, and test it on the test set and record the results of the test (ACC).
