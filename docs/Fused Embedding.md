# Fused Embedding

## 介绍

DeepRec 及 TensorFlow 原生的 embedding lookup 相关 API，如 safe_fused_embedding_lookup_sparse，会创建比较多的 op，因此在 GPU 上执行时容易出现 kernel launch bound 的问题。因此，fused_embedding 将其中一部分算子融合，达到在 GPU 上加速执行的目的。
## 使用方法
目前在 `tensorflow/python/feature_column/feature_column_v2.py` 的`EmbeddingColumn` 及`tensorflow/contrib/layers/python/layers/feature_column.py` 的`_EmbeddingColumn` 类中都增加了 `use_fused_lookup` 的选项。因此只要用到这两个类作为 embedding column, 则在 lookup 时都可以选择使用 fused_lookup。使用例子：
​


1. 使用`categorical_column_with_embedding`接口：
```python
import tensorflow as tf
from tensorflow.python.framework import ops


columns = tf.feature_column.categorical_column_with_embedding("col_emb", dtype=tf.dtypes.int64)
W = tf.feature_column.embedding_column(categorical_column=columns,
            dimension=3,
            initializer=tf.ones_initializer(tf.dtypes.float32))

# set use fused_lookup
W.use_fused_lookup = True

ids={}
ids["col_emb"] = tf.SparseTensor(indices=[[0,0],[1,1],[2,2],[3,3],[4,4]], values=tf.cast([1,2,3,4,5], tf.dtypes.int64), dense_shape=[5, 4])

# There is internal logic that will use fused_lookup for W
emb = tf.feature_column.input_layer(ids, [W])
fun = tf.multiply(emb, 2.0, name='multiply')
loss = tf.reduce_sum(fun, name='reduce_sum')
opt = tf.train.FtrlOptimizer(0.1, l1_regularization_strength=2.0, l2_regularization_strength=0.00001)
g_v = opt.compute_gradients(loss)
train_op = opt.apply_gradients(g_v)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print("init global done")
    print(sess.run([emb, train_op,loss]))
    print(sess.run([emb, train_op,loss]))
    print(sess.run([emb, train_op,loss]))
```

2. 使用`sparse_column_with_embedding`接口：
```python
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.contrib.layers.python.layers import feature_column_ops
from tensorflow.contrib.layers.python.layers import feature_column


columns = feature_column.sparse_column_with_embedding(column_name="col_emb", dtype=tf.dtypes.int64)
W = feature_column.embedding_column(sparse_id_column=columns,
            dimension=3,
            initializer=tf.ones_initializer(tf.dtypes.float32))

# set use fused_lookup
W.use_fused_lookup = True

ids={}
ids["col_emb"] = tf.SparseTensor(indices=[[0,0],[1,1],[2,2],[3,3],[4,4]], values=tf.cast([1,2,3,4,5], tf.dtypes.int64), dense_shape=[5, 4])

# There is internal logic that will use fused_lookup for W
emb = feature_column_ops.input_from_feature_columns(columns_to_tensors=ids, feature_columns=[W])
fun = tf.multiply(emb, 2.0, name='multiply')
loss = tf.reduce_sum(fun, name='reduce_sum')
opt = tf.train.FtrlOptimizer(0.1, l1_regularization_strength=2.0, l2_regularization_strength=0.00001)
g_v = opt.compute_gradients(loss)
train_op = opt.apply_gradients(g_v)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print("init global done")
    print(sess.run([emb, train_op,loss]))
    print(sess.run([emb, train_op,loss]))
    print(sess.run([emb, train_op,loss]))
```


他们在内部，实际上都调用 `tensorflow/contrib/layers/python/layers/embedding_ops.py` 中的 `safe_fused_embedding_lookup_sparse` 方法，这个方法功能上和 `safe_embedding_lookup_sparse` 是相同的
## 注意事项

1. 目前 fused embedding lookup 算子必须是在 Nvidia GPU 上才可执行。相应的 tf.Variable 和 EmbeddingVariable 及其他算子可以在 CPU 上。
1. 如果使用了 scattered_embedding_column(hash_key 方式查找), 是不支持  fused embedding lookup 的。
1. safe_embedding_lookup_sparse 中有设置 sparse_weights 的功能，safe_fused_embedding_lookup_sparse 还不支持。
1. partition_strategy 目前只支持 div 的模式
1. 使用 blocknums（给 DynamicEmbeddingVariable）还不支持
1. 目前 fused 部分覆盖了相当于 tensorflow/python/ops/embedding_ops.py 的 embedding_lookup_sparse 中的内容。在之前的一些维度拍平，缺失值补充等功能还未 fuse。
