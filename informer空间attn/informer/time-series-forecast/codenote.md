
## model

```python
# informer.py
class InformerModel(BaseModel):

    def create_model(self):
        """ 分bin构建多个model """
        return [self.create_informer_model() for i in range(len(self.rank_bins) - 1)]
    def create_informer_model(self):
        """ 构建Model """
        seq2seq_model = tf.keras.models.Model(
            [encoder_dense_input, encoder_sparse_input, encoder_pos_input, decoder_dense_input, decoder_sparse_input, decoder_pos_input],
            output)
        return seq2seq_model


    def fit_model(self, _bin, train_x, train_y, train_w, val_x, val_y, val_w):
        """ 
        input: [(651, 60, 14), (651, 60, 5), (651, 60, 1), (651, 60, 8), (651, 60, 5), (651, 60, 1)]
        output: (651, 30, 1) 第一个维度也是630+21
        """
        self.models[_bin].fit(
            list(map(lambda x: np.append(x[0], values=x[1], axis=0), zip(train_x, val_x))), 
            np.append(train_y, values=val_y, axis=0), 
            ...
        )

```

