

- 20+21AAAI+Informer
    - Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting
    - tags: #Attention; #LSTF; #Transformer; #ProbSparse; #Informer
    - [arxiv](https://arxiv.org/abs/2012.07436); [github](https://github.com/zhouhaoyi/Informer2020)
    - [codenote](https://mp.weixin.qq.com/s/hZppLj3eR-aql9w3M-vjKg)
    - abs: 针对长时序预测问题 Long sequence time-series forecasting (LSTF). Transformer 的问题: quadratic time complexity, high memory usage, and inherent limitation of the encoder-decoder architecture. 因此 Informer 针对性地 1] ProbSparse self-attention mechanism 可以达到 O(L logL) 的 time complexity and memory usage. 2] 从而能够处理超长的输入; 3] generative style decoder 直接一次不需要序列生成, 加速. 

## code

```python
class Exp_Informer(Exp_Basic):

    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
        # batch_x [32,96,7], batch_y [32,72,7]
        # 时间维度 batch_x_mark [32,96,4], batch_y_mark [32,72,4]
        
        # dec_inp: [32,72,7]
        dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).float().to(self.device)
        
        # outputs: [32,24,7]
        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        
        # batch_y [32,72,7]
        f_dim = -1 if self.args.features=='MS' else 0
        batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)
        return outputs, batch_y
```


### Informer

```python
class Informer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, ):
        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers-1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, ):
        # enc_out: [32,96,512] 其中的 x_enc [B,L,7], x_mark_enc [B,L,4]
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        # dec_out: [32, 72, 512]
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        # 512 -> 7
        dec_out = self.projection(dec_out)
        return dec_out[:,-self.pred_len:,:] # [B, L, D]
```

### Attention

```python
class AttentionLayer(nn.Module):

```


### Encoder

```python
class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):

    def forward(self, x, attn_mask=None):
        # x [B, L, D] = [32, 96, 512]
        for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
            x, attn = attn_layer(x, attn_mask=attn_mask)
            # [32, 96, 512] => [32, 48, 512]
            x = conv_layer(x)
            attns.append(attn)
        x, attn = self.attn_layers[-1](x, attn_mask=attn_mask)
        attns.append(attn)



class ConvLayer(nn.Module):
    def forward(self, x):
        # x: [32, 96, 512] -> [32, 48, 512]
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1,2)
        return x
```


### Decoder

```python
class Decoder(nn.Module):
    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
    def forward(self, x, cross, x_mask=None, cross_mask=None):
        # x: [32, 72, 512]
        # cross: [32, 48, 512]
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])

        # y: [32, 72, 512] => [32, 72, 512] 中间层 512 -> 2048 -> 512
        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))

        return self.norm3(x+y)
```



