# BERT微调(RNN文本分类)
本模型下游任务是RNN文本分类，用bert生成词向量
编写RNN的语言模型，并基于训练好的词向量，编写RNN模型用于文本分类 (参考文献如下)

Yang, Zichao, Diyi Yang, Chris Dyer, Xiaodong He, Alex Smola, and Eduard Hovy. "Hierarchical attention networks for document classification." In *Proceedings of the 2016 conference of the North American chapter of the association for computational linguistics: human language technologies*, pp. 1480-1489. 2016.

1. **网络框架**：pytorch
2. **数据集**：使用 Yelp2013 数据集。使用数据集中的test.json当作测试集，并从yelp_academic_dataset_review.json中手动划分训练集和验证集。下载链接：https://github.com/rekiksab/Yelp/tree/master/yelp_challenge/yelp_phoenix_academic_dataset 只使用stars评分和text评论内容。
3. **模型搭建**：采用 pytorch 或所封装的 module编写模型，torch.nn.Linear(), torch.nn.Relu() 等。
4. **模型训练**：将生成的训练集输入搭建好的模型进行前向的 loss 计算和反向的梯度传播，从而训练模型，同时也建议使用网络框架封装的 optimizer 完成参数更新过程。训练过程中记录模型在训练集和验证集上的损失，并绘图可视化。
5. **调参分析**：将训练好的模型在验证集上进行测试，以 **Top 1 Accuracy(ACC)** 作为网络性能指标。然后，对 dropout, normalization, learning rate decay, residual connection, network depth 进行调整，再重新训练、测试，并分析对模型性能的影响。
6. **测试性能**：选择在验证集上表现最好的一组超参数，重新训练模型，并在测试集上测试并记录测试的结果（ACC）。