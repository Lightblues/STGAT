

## model

```python
class STInformer(tf.keras.layers.Layer):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, batch_size,

    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, out_len_start, batch_size,
                 factor=1, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', activation='gelu', have_enc=True):

    def call(self, inputs, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        x_enc, x_dec = inputs
        enc_out = None
        if self.have_enc:
            enc_out = self.encoder(x_enc, attn_mask=enc_self_mask)

        dec_out = self.decoder(x_dec, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        dec_out = self.projection(dec_out)

        return dec_out[:, :, (-self.pred_len+self.pred_len_start-1):, :]  # [B, L, D]


```

encoder

```python
class EncoderLayer(Layer):
    def call(self, x, attn_mask=None):
        # x [B, X, L, D]
        x = x + self.dropout(self.t_attention(
            [x, x, x],
            attn_mask = attn_mask
        ))
        x = self.norm1_1(x)
        x = x + self.dropout(self.s_attention(
            [x, x, x],
            attn_mask = attn_mask
        ))
        y = x = self.norm1_2(x)

        y = self.dropout(self.activation(self.conv1(y)))
        y = self.dropout(self.conv2(y))
#        return self.norm2(x+y)
        return x + self.norm2(y)
```

## data

```python
class BaseModel:
    def read_data(self):
        logger.info(">>>>>>>>>>Start reading data:")
        all_cols = list(...)
        logger.info(">>>>>>>>>>All columns: {}".format(all_cols))
        ## rank limit
        logger.info(">>>>>>>>>>Rank limits from {} to {}".format(rank_lower_limit, rank_upper_limit))
        ## read data
            dataset = pd.read_csv(self.args.data_path)
            self.data_date = dataset['d'][0]
            dataset = dataset[
                (dataset[self.args.rank_col] <= rank_upper_limit) & (dataset[self.args.rank_col] >= rank_lower_limit)][
                all_cols]
        logger.info(">>>>>>>>>>Hive data date: {}".format(self.data_date))


```


