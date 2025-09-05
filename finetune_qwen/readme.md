##
Trie类实现前缀树，用来约束生成过程。避免生成不存在的item。

ConstrainedCLMTrainer类继承huggingface的Train类。实现在训练过程中的评估阶段的前缀树搜索和reall、ndcg指标的计算。

416行到470行为数据处理，得到映射为码本后的序列数据。

再往下是训练的代码