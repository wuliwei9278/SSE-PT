# SSE-PT: Temporal Collaborative Ranking Via Personalized Transformer
We implement our code in Tensorflow and the code is tested under a server with 40-core Intel Xeon E5-2630 
v4 @ 2.20GHz CPU, 256G RAM and Nvidia GTX 1080 GPUs (with TensorFlow 1.13 and Python 3).

## Datasets
The preprocessed datasets are in the `data` directory (`e.g. data/ml1m.txt`). Each line of the `txt` format data contains
a `user id` and an `item id`, where both user id and item id are indexed from 1 consecutively. Each line represents one interaction between the user 
and the item. For every user, their interactions were sorted by timestamp.

## Options
The training of the SSE-PT model is handled by the main.py script that provides the following command line arguments.
```
--dataset            STR           Name of dataset.               Default is "ml1m".
--train_dir          STR           Train directory.               Default is "default".
--batch_size         INT           Batch size.                    Default is 128.    
--lr                 FLOAT         Learning rate.                 Default is 0.001.
--maxlen             INT           Maxmum length of sequence.     Default is 50.
--user_hidden_units  INT           Hidden units of user.          Default is 50.
--item_hidden_units  INT           Hidden units of item.          Default is 50.
--num_blocks         INT           Number of blocks.              Default is 2.
--num_epochs         INT           Number of epochs to run.       Default is 2001.
--num_heads          INT           Number of heads.               Default is 1.
--dropout_rate       FLOAT         Dropout rate value.            Default is 0.5.
--threshold_user     FLOAT         SSE probability of user.       Default is 1.0.
--threshold_item     FLOAT         SSE probability of item.       Default is 1.0.
--l2_emb             FLOAT         L2 regularization value.       Default is 0.0.
--gpu                INT           Name of GPU to use.            Default is 0.
--print_freq         INT           Print frequency of evaluation. Default is 10.
--k                  INT           Top k for NDCG and Hits.       Default is 10.
```
## Commands
To train our model on the default `ml1m` data with default parameters:
```
python3 main.py
``` 
To train a SSE-PT model on `ml1m` data using a maxlen of 200, a dropout rate of 0.2, a SSE probability of 0.08 for user side 
and a SSE probability of 0.9 for item side.
```
python3 main.py --maxlen=200 --dropout_rate 0.2 --threshold_user 0.08 --threshold_item 0.9
```

## Results
The following is the plot of NDCG@10 versus training time (Seconds) for SASRec, SSE-PT and SSE-PT++. Our proposed SSE-PT and SSE-PT++ outperform SASRec.
<p align="center">
  <img width="600" src="ml1m_speed.png">
</p>

