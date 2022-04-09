# GLUE_benchmark_dataset_split
 Use the dev set as the test set for GLUE datasets except SST-2.
 Then, split the training set in the GLUE dataset into training and dev sets.

Use the "download_dataset.sh" to download tha dataset. Afterwards, make the "original" folder in the root directory of this repo.
>sampled_from_train_xxx means the number of data sampled from training set as dev set
> Default parameters mean "len(dev_set) = len(test_set)"

``````
python3 make_file.py \
        --seed 19   \
        --copy True \
        --sampled_from_train_CoLA 1043 \
        --sampled_from_train_MRPC 408 \
        --sampled_from_train_STS_B 1500 \
        --sampled_from_train_QQP 40430 \
        --sampled_from_train_MNLI 9832 \
        --sampled_from_train_QNLI 5463 \
        --sampled_from_train_RTE 277 \
        --sampled_from_train_WNLI 71
