# -*- coding: utf-8 -*-
"""
This script contains the layers used by Transformers, that can be called to
contruct a model.

Based on: https://machinelearningmastery.com/implementing-the-transformer-encoder-from-scratch-in-tensorflow-and-keras/
"""
#%% Import libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras import backend as K
 
#%% Transformers modules

class DotProductAttention(layers.Layer):
    
    def __init__(self, scale_factor: int, return_sequences: bool = False, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.scale_factor = scale_factor
        self.return_sequences = return_sequences
 
    def call(self, query, value, key=None, attention_mask=None, return_scores=False):
        if key is None:
            key = value
        
        # Scoring the queries against the keys + scaling
        scores = tf.matmul(query, key, transpose_b=True) # scores.shape=(d_q,d_k)
        scores = scores / tf.math.sqrt(tf.cast(self.scale_factor, tf.float32))
 
        # Apply mask to the attention scores
        if attention_mask is not None:
            attention_mask = tf.cast(attention_mask, tf.float32) # because it's boolean mask
            attention_mask = 1e9 * (attention_mask-1) # masked values turn (-inf)
            scores += tf.expand_dims(attention_mask,-2) + tf.expand_dims(attention_mask,-1)
 
        # Computing the weights by a softmax operation
        weights = K.softmax(scores) # attention weights!
 
        # Computing the attention by a weighted sum of the value vectors
        context = tf.matmul(weights, value) # out.shape=(d_q,dim)
        
        # Define output
        if not self.return_sequences:
            context = K.sum(context, axis=1)
        
        if return_scores:
            return context, weights
        else:
            return context
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "scale_factor": self.scale_factor,
            "return_sequences": self.return_sequences,
        })
        return config
    
class LocalAttention(layers.Layer):
    def __init__(self, dim_FFNN1: int, 
                       d_model: int, 
                       drop_rate: float,
                       activation: str,
                       **kwargs):
        super(LocalAttention, self).__init__(**kwargs)
        self.dim_FFNN1 = dim_FFNN1
        self.d_model = d_model
        self.drop_rate = drop_rate
        self.activation = activation
        
        self.MLP_Att = \
        Sequential([layers.Dense(dim_FFNN1, activation=activation),
                    layers.Dropout(drop_rate),
                    layers.Dense(d_model, activation=None),
                    layers.Activation('softmax')
                    ], name="MLP")
        
    def call(self, inputs, return_scores=False):
        weights = self.MLP_Att(inputs)
        context = tf.matmul(weights, inputs, transpose_b=True)
        if return_scores:
            return context, weights
        return context
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "dim_FFNN1": self.dim_FFNN1,
            "d_model": self.d_model,
            "drop_rate": self.drop_rate,
            "activation": self.activation,
        })
        return config

# Implementing the Multi-Head Attention
class MHAttention(layers.Layer):
    def __init__(self, heads: int, 
                       dim_K: int, 
                       dim_V: int, 
                       d_model: int, 
                       activation: str = None, 
                       **kwargs):
        super(MHAttention, self).__init__(**kwargs)
        assert heads*dim_K == d_model, \
            "Invalid parameters: d_model must be equal to heads*dim_K,"\
            " but got\n heads*dim_K: {}\n d_model: {}".format(heads*dim_K,d_model)
        
        self.heads = heads  # Number of attention heads to use
        self.dim_K = dim_K  # Dimensionality of the linearly projected queries and keys
        self.dim_V = dim_V  # Dimensionality of the linearly projected values
        self.d_model = d_model  # Dimensionality of the model
        self.activation = activation
        
        # Layers for linear projection from d_model to d_k/d_v (INPUT)
        self.Wq = layers.Dense(dim_K*heads, activation=None)  # Projection matrix for queries
        self.Wq_Act = layers.Activation(activation)
        self.Wk = layers.Dense(dim_K*heads, activation=None)  # Projection matrix for keys
        self.Wk_Act = layers.Activation(activation)
        self.Wv = layers.Dense(dim_V*heads, activation=None)  # Projection matrix for values
        self.Wv_Act = layers.Activation(activation)
        
        # Scaled dot product attention
        self.attention = DotProductAttention(dim_K, return_sequences=True)
        
        # Layer for linear projection from d_k/d_v to d_model (OUTPUT)
        self.Wo = layers.Dense(d_model, activation=None)  # Projection matrix for multi-head
        self.Wo_Act = layers.Activation(activation)
        
    def divide_heads(self, x, flag):
        """
        Allows the attention head to be computed in parallel
         Expects x of shape=(batch_size, max_seq_len, d_k/v)
        """
        if flag:
            # Tensor shape after reshaping and transposing: (batch_size, heads, seq_length, depth)
            # special value -1 will fit the rest of the shape into that dimension
            x = tf.reshape(x, shape=(tf.shape(x)[0], tf.shape(x)[1], self.heads, -1))
            x = tf.transpose(x, perm=[0, 2, 1, 3])
        else:
            # Reverting the reshaping and transposing operations: (batch_size, seq_length, d_k)
            # This perform the operation of concatenation
            x = tf.transpose(x, perm=[0, 2, 1, 3])
            x = tf.reshape(x, shape=(tf.shape(x)[0], tf.shape(x)[1], self.dim_K*self.heads))
        return x
        # return tf.reshape(x, (-1, x.shape[-2], x.shape[-1]) )
 
    def call(self, query, value, key=None, attention_mask=None, return_attention_scores=False):
        """
        Expects queries, keys and values with shapes (respectively):
            (batch_size, max_seq_len, d_model)
            (batch_size, max_seq_len, d_model)
            (batch_size, max_seq_len, d_model)
        d_model is the dimensionality of the embeddings
        """
        if key is None:
            key = value
        
        # Linear projection of Q, K and V
        query_D = self.Wq(query)
        query_D = self.Wq_Act(query_D)
        value_D = self.Wv(value)
        value_D = self.Wv_Act(value_D)
        key_D   = self.Wk(key)
        key_D   = self.Wk_Act(key_D)
        # Reshape Q, K and V to enter the dot-product attention in parallel
        query_H = self.divide_heads(query_D, True)
        value_H = self.divide_heads(value_D, True)
        key_H   = self.divide_heads(key_D,   True)
        # Each resulting tensor shape: (batch_size, heads, input_seq_length, Dh=D/H)
    
        # Compute multi-head attention output, parallely, using reshaped Q, K and V
        if attention_mask is not None:
            attention_mask = attention_mask[:,tf.newaxis, :] # add attention dimension

        if return_attention_scores:
            output_H, Att_scores = self.attention(query_H, value_H, key_H, 
                                                  attention_mask=attention_mask,
                                                  return_scores=return_attention_scores)
        else:
            output_H = self.attention(query_H, value_H, key_H,
                                      attention_mask=attention_mask)
        # Rearrange back the output into concatenated form
        output_D = self.divide_heads(output_H, False)
        
        context = self.Wo(output_D) # shape: (batch_size, input_seq_length, d_model)
    
        if return_attention_scores:
            return self.Wo_Act(context), Att_scores
        else:
            return self.Wo_Act(context)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "heads": self.heads,
            "dim_K": self.dim_K,
            "dim_V": self.dim_V,
            "d_model": self.d_model,
            "activation": self.activation,
        })
        return config
    
# Implementing the Add & Norm Layer
class AddNormalization(layers.Layer):
    def __init__(self, **kwargs):
        super(AddNormalization, self).__init__(**kwargs)
        self.layer_norm = layers.LayerNormalization()  # Layer normalization layer
 
    def call(self, x, sublayer_x):
        # The sublayer input and output need to be of the same shape to be summed
        add = x + sublayer_x
 
        # Apply layer normalization to the sum
        return self.layer_norm(add)
 
# Implementing the Feed-Forward Layer
class FeedForward(layers.Layer):
    def __init__(self, dim_FFNN: int, d_model: int, **kwargs):
        super(FeedForward, self).__init__(**kwargs)
        self.dim_FFNN = dim_FFNN
        self.d_model = d_model

        self.fully_connected1 = layers.Dense(dim_FFNN, activation=None)
        self.activation = layers.ReLU()
        self.fully_connected2 = layers.Dense(d_model)
 
    def call(self, x):
        # The input is passed into the two fully-connected layers, with a ReLU in between
        fully_connected1_out = self.fully_connected1(x)
        relu_out = self.activation(fully_connected1_out)
        fully_connected2_out = self.fully_connected2(relu_out)
 
        return fully_connected2_out

    def get_config(self):
        config = super().get_config()
        config.update({
            "dim_FFNN": self.dim_FFNN,
            "d_model": self.d_model,
        })
        return config
    
# Implementing the Token and Positional Embedding Layer
class TokenAndPositionEmbedding(layers.Layer):
    
    def __init__(self, vocab_size: int,
                       embed_dim: int, 
                       maxlen: int,
                       weights_tokens: np.array = None,
                       weights_position: np.array = None, 
                       mask_zero: bool = False, 
                       **kwargs):
        super(TokenAndPositionEmbedding, self).__init__(**kwargs)
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.maxlen = maxlen
        self.weights_tokens = weights_tokens
        self.weights_position = weights_position
        self.mask_zero = mask_zero
        
        # Define embedding layer
        if isinstance(weights_tokens, np.ndarray):
            assert self._verify_tok_weights_shape(weights_tokens), \
                "Invalid shape of weights_tokens: Expected {}, but got {}."\
                    .format((self.vocab_size,self.embed_dim),weights_position.shape)
            wT = [weights_tokens] 
        else: wT = None
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim,
                                          mask_zero=mask_zero,
                                          weights=wT)
        if isinstance(weights_position, np.ndarray):
            assert self._verify_pos_weights_shape(weights_position), \
                "Invalid shape of weights_position: Expected {}, but got {}."\
                    .format((self.maxlen,self.embed_dim),weights_position.shape)
            self.pos_emb = tf.constant(weights_position, dtype="float32")
        else:
            self.pos_emb = tf.random.normal((maxlen,embed_dim), dtype="float32")

    def call(self, inputs):
        """Input tokens convert into embedding vectors then superimposed
        with position vectors"""
        embedded_tokens = self.token_emb(inputs)
        return embedded_tokens + self.pos_emb
    
    # From:https://github.com/keras-team/keras/blob/v2.11.0/keras/layers/core/embedding.py#L34
    def compute_mask(self, *args, **kwargs):
        return self.token_emb.compute_mask(*args, **kwargs)
    
    def _verify_tok_weights_shape(self, W):
        Wshape = W.shape
        return Wshape[0]==self.vocab_size and Wshape[1]==self.embed_dim
        
    def _verify_pos_weights_shape(self, W):
        Wshape = W.shape
        return Wshape[0]==self.maxlen and Wshape[1]==self.embed_dim
        

    def get_config(self):
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "maxlen": self.maxlen,
            "weights_tokens": self.weights_tokens,
            "weights_position": self.weights_position,
            "mask_zero": self.mask_zero, 
        })
        return config
    
def build_pos_matrix(L, d, n=10000):
    """Create positional encoding matrix

    Args:
        L: Input dimension (length)
        d: Output dimension (depth), even only
        n: Constant for the sinusoidal functions

    Returns:
        numpy matrix of floats of dimension L-by-d. At element (k,2i) the value
        is sin(k/n^(2i/d)) while at element (k,2i+1) the value is cos(k/n^(2i/d))
    """
    assert d % 2 == 0, "Output dimension needs to be an even integer"
    d2 = d//2
    P = np.zeros((L, d))
    k = np.arange(L).reshape(-1, 1)     # L-column vector
    i = np.arange(d2).reshape(1, -1)    # d-row vector
    denom = np.power(n, -i/d2)          # n**(-2*i/d)
    args = k * denom                    # (L,d) matrix
    P[:, ::2] = np.sin(args)
    P[:, 1::2] = np.cos(args)
    return P
    
class Encoder(layers.Layer):
    
    def __init__(self, heads: int, 
                       ff_dim: int, 
                       key_dim: int, 
                       val_dim: int, 
                       model_dim: int, 
                       drop_rate: float, 
                       **kwargs):
        """
        This Layer performes one encoder inspired on the Transformer. The 
        encoder is formed by a multihead attention block and and feed forward
        neural network.
        
        Note: The Transformer uses N=6 Encoders stacked for the model's encoder.
        
        Parameters:
            heads (int): number of attention heads to use in the multihead attention
             layer. In The Transformer, its value is 8.
             
            key_dim (int): number of dimensions of the projection of the queries
             and keys. As this is the encoder, querie and key are copies of the 
             same input. In The Transformer, its value is 64.
             
            val_dim (int): number of dimensions of the projection of the values.
             In The Transformer, its value is 64.
             
            model_dim (int): number of dimensions of the model. It is the dimensionality
             of the embeddings (token and position) and of all sub-layers outputs.
             In The Transformer, its value is 512.
             
            ff_dim (int): number of units of the inner Linear transformation of
             feed forward neural network. In The Transformer, its value is 2048.
            
            drop_rate (float): probability of dropout layers. In The Transformer,
             its value is 0.10.
        """
        super(Encoder, self).__init__(**kwargs)
        
        self.multihead_attention = MHAttention(heads, key_dim, val_dim, 
                                               model_dim, 'relu')
        self.dropout1 = layers.Dropout(drop_rate)
        self.add_norm1 = AddNormalization()
        self.ffnn = FeedForward(ff_dim, model_dim)
        self.dropout2 = layers.Dropout(drop_rate)
        self.add_norm2 = AddNormalization()
        
        """
        When using MultiHeadAttention (from keras) inside a custom Layer, 
        the custom Layer must implement `build()` and call MultiHeadAttention's 
        `_build_from_signature()`.
        This enables weights to be restored correctly when the model is loaded.
        TODO(b/172609172): link to documentation about calling custom build functions
        when used in a custom Layer.
        """
        
    def call(self, query, value, key=None, mask=None, training=None):
        if key is None:
            key = value
        
        multihead_out = self.multihead_attention(query, value, key, attention_mask=mask)
        multihead_out = self.dropout1(multihead_out, training=training)
        addnorm_out = self.add_norm1(query, multihead_out)
        
        ffnn_out = self.ffnn(addnorm_out)
        ffnn_out = self.dropout2(ffnn_out, training=training)
        encoder_out = self.add_norm2(multihead_out, ffnn_out)
        
        return encoder_out

#%% END
if __name__ == '__main__': # testing examples
    #%% Parameters
    batch_size = 2
    vocab_size = 1000
    input_seq_length = 5
    d_k = 15
    d_v = 15
    d_model = 30
    h = 2
    ff_dim = 60
    
    #%% Test TokenAndPositionEmbedding
    input_seq = np.random.randint(1, vocab_size, (batch_size, input_seq_length))
    input_seq[:,-2:] = 0
    mask = np.ones((batch_size, input_seq_length))
    mask[:,3:] = 0.
    embedding_matrix = np.random.normal(size=(vocab_size,d_model))
    embedding_matrix[0,:] = np.ones((1,d_model))*0.001
    position_matrix = build_pos_matrix(input_seq_length,
                                      d_model)
    
    Embed_layer = TokenAndPositionEmbedding(vocab_size,d_model,
                                            input_seq_length,
                                            weights_tokens=embedding_matrix,
                                            weights_position=position_matrix,
                                            mask_zero=True
                                            )
    
    embed_input = Embed_layer(input_seq)
    mask_input = Embed_layer.compute_mask(input_seq)
    print(embed_input, '\n')
    print(mask_input, '\n')
    
    #%% Test DotProductAttention
    queries = np.random.random((batch_size, input_seq_length, d_model))
    keys = np.random.random((batch_size, input_seq_length, d_model))
    values = np.random.random((batch_size, input_seq_length, d_model))
    mask = np.ones((batch_size, input_seq_length))
    mask[:,3:] = 0.
    
    dot_attention = DotProductAttention(d_model, True)
    
    dot_att, scores = dot_attention(queries, values, keys, 
                                    attention_mask=mask,
                                    return_scores=True)
    print(dot_att, '\n')
    print(scores, '\n')
    
    #%% Test LocalAttention
    inputs = np.random.random((batch_size, input_seq_length, d_model))
    local_attention = LocalAttention(dim_FFNN1=d_k,
                                     d_model=d_model, 
                                     drop_rate=0.3,
                                     activation='relu')
    local_att,w = local_attention(inputs, return_scores=True)
    print(local_att, '\n')
    
    #%% Test MHAttention
    queries = np.random.random((batch_size, input_seq_length, d_model))
    keys = np.random.random((batch_size, input_seq_length, d_model))
    values = np.random.random((batch_size, input_seq_length, d_model))
    mask = np.ones((batch_size, input_seq_length))
    mask[:,3:] = 0.
    
    multihead_attention = MHAttention(h, d_k, d_v, d_model,'relu')
    
    multi_att = multihead_attention(queries, values, keys,
                                    attention_mask=mask)
    print(multi_att, '\n')
    
    #%% Test AddNormalization
    queries = np.random.random((batch_size, input_seq_length, d_model))
    # (...)
    output = np.random.random((batch_size, input_seq_length, d_model))
    
    add_norm = AddNormalization()
    
    addnorm = add_norm(output, queries)
    print(addnorm, '\n')
    
    #%% Test FeedForward
    output = np.random.random((batch_size, input_seq_length, d_model))
    
    ffnn = FeedForward(ff_dim, d_model)
    
    ff_out = ffnn(output)
    print(ff_out, '\n')
    
    #%% Test Encoder
    inputs = np.random.random((batch_size, input_seq_length, d_model))
    mask = np.ones((batch_size, input_seq_length))
    mask[:,3:] = 0.
    
    enc = Encoder(h, ff_dim, d_k, d_v, d_model, 0.1)
    
    context = enc(inputs, inputs, mask=mask)
    print(context, '\n')
    
    #%% All together
    """
    All isolated tests ran OK. With and without passing
    padding mask argument.
    """
    batch_size = 128
    vocab_size = 11560
    max_seq_len = 128
    d_k = 64
    d_v = d_k
    h = 8
    vector_size = 512
    internalFFNN = 1024
    embedding_matrix = np.random.normal(size=(vocab_size,vector_size))
    position_matrix = build_pos_matrix(max_seq_len,vector_size)
    
    # Define layers
    model_input = layers.Input((max_seq_len,))
    Embed_layer = TokenAndPositionEmbedding(vocab_size=vocab_size,
                                            embed_dim=vector_size,
                                            maxlen=max_seq_len,
                                            weights_tokens=embedding_matrix,
                                            weights_position=position_matrix,
                                            mask_zero=True)
    
    Encoder_layer = Encoder(heads=h, ff_dim=internalFFNN, 
                        key_dim=d_k, val_dim=d_v, model_dim=vector_size, 
                        drop_rate=0.1)
    
    embed_input = Embed_layer(model_input)
    input_mask = Embed_layer.compute_mask(model_input)
    context = Encoder_layer(embed_input, embed_input, mask=input_mask)
    
    classifier_out = layers.Dense(3,activation=None)(context)
    model_output = layers.Activation('softmax')(classifier_out)
    
    model = tf.keras.Model(model_input, model_output)
    model.summary()
    
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
    loss_func = tf.keras.losses.CategoricalCrossentropy()
    metrics = ['accuracy','AUC']
    
    model.compile(optimizer=optimizer, loss=loss_func, metrics=metrics)
    model.save("./Transformers/Model")










