
## model

### InformerModel

debug技巧: Keras模型设置 `seq2seq_model.run_eagerly = True`, 这样就可以在fit的时候打断点了 (进入调试)
[doc](https://keras.io/examples/keras_recipes/debugging_tips/)

```python
# informer.py
class InformerModel(BaseModel):

    def create_model(self):
        """ 分bin构建多个model """
        return [self.create_informer_model() for i in range(len(self.rank_bins) - 1)]
    def create_informer_model(self):
        """ 构建Model """
        transformer_layer = Informer(enc_in=512, dec_in=512, c_out=1, seq_len=self.encoder_timesteps, label_len=0,
                                     out_len=self.decoder_timesteps, batch_size=self.model_params['batch_size'],
                                     factor=5, d_model=512, n_heads=8, e_layers=1, d_layers=1, d_ff=512,
                                     dropout=0.2, attn=self.model_params['attn_type'], embed='fixed', data='ETTh', activation='gelu')

        output = transformer_layer([encoder_input, decoder_input])

        seq2seq_model = tf.keras.models.Model(
            [encoder_dense_input, encoder_sparse_input, encoder_pos_input, decoder_dense_input, decoder_sparse_input, decoder_pos_input],
            output)
        seq2seq_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.model_params['learning_rate']),
                              loss=self.model_params['loss'],
                              metrics=['MSE', 'MAE', 'MAPE']
                              )
        # for debug
        seq2seq_model.run_eagerly = True
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


### Informer

```python
class Informer(tf.keras.layers.Layer):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, batch_size,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', embed='fixed', data='ETTh', activation='gelu', have_enc=True):
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout),
                                   d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=tf.keras.layers.LayerNormalization()
        )

    def call(self, inputs, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        x_enc, x_dec = inputs
        if self.have_enc:
            enc_out = self.encoder(x_enc, attn_mask=enc_self_mask)
        dec_out = self.decoder(x_dec, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        dec_out = self.projection(dec_out)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]

```

