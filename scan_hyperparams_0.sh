#!/bin/bash
param1="--max_lr=0.002 --num_epochs=5 --iter_name=bi_lstm_gru_maxLR2_ep5"
param2="--max_lr=0.003 --num_epochs=5 --iter_name=bi_lstm_gru_maxLR3_ep5"
param3="--max_lr=0.002 --num_epochs=5 --iter_name=bi_gru_maxLR2_ep5_h60 --model_name=bi_gru --hidden_nodes=60"
param4="--max_lr=0.002 --num_epochs=2 --iter_name=bi_lstm_maxLR2_ep5_glove_new --train_embeddings=True --model_name=bi_lstm"
param5="--max_lr=0.002 --num_epochs=2 --iter_name=bi_lstm_maxLR2_ep5_glove --model_name=bi_lstm"
param6="--num_epochs=5 --iter_name=bi_lstm_retrGloveWiki_ep5 --train_embeddings=True --model_name=bi_lstm --use_wiki=True --use_pooling=True --use_attention=True"
param7="--num_epochs=5 --iter_name=bi_lstm_retrGlove_ep5 --train_embeddings=True --model_name=bi_lstm --use_pooling=True --use_attention=True"
param8="--num_epochs=5 --iter_name=bi_lstm_retrGlove_ep5_len90 --train_embeddings=True --model_name=bi_lstm --maxlen=90 --use_pooling=True --use_attention=True"
param9="--num_epochs=2 --iter_name=bi_lstm_retrGlove_ep2_redType1 --train_embeddings=True --model_name=bi_lstm --reduction_dim_type=1"
param10="--num_epochs=2 --iter_name=bi_lstm_retrGlove_ep2_redType2 --train_embeddings=True --model_name=bi_lstm --reduction_dim_type=2"
param11="--num_epochs=2 --iter_name=bi_lstm_retrGlove_ep2_redType3 --train_embeddings=True --model_name=bi_lstm --reduction_dim_type=3"
param12="--num_epochs=2 --iter_name=bi_lstm_retrGlove_ep2_redType3_h64 --train_embeddings=True --model_name=bi_lstm --reduction_dim_type=3 --hidden_nodes=64"
param13="--num_epochs=3 --iter_name=bi_lstm_retrGlove_ep3_redType3_h64_mL005 --train_embeddings=True --model_name=bi_lstm --reduction_dim_type=3 --hidden_nodes=64 --max_lr=0.005"
param14="--num_epochs=3 --iter_name=bi_lstm_retrGlove_ep3_redType1_h64 --train_embeddings=True --model_name=bi_lstm --reduction_dim_type=1 --hidden_nodes=64"
param15="--num_epochs=3 --iter_name=bi_lstm_vocab50_retrGlove_ep3_redType1_h64 --train_embeddings=True --model_name=bi_lstm --reduction_dim_type=1 --hidden_nodes=64 --vocab_size=50000"
param16="--num_epochs=3 --iter_name=bi_lstm_CV4_retrGlove_ep3_redType1_h64 --train_embeddings=True --model_name=bi_lstm --reduction_dim_type=1 --hidden_nodes=64 --cv_folds=4"
#---- new version --> converted booleans to integers ( 1 / 0)
param17="--num_epochs=2 --iter_name=bi_lstm_base_model --train_embeddings=1 --model_name=bi_lstm --reduction_dim_type=3 --hidden_nodes=64 --cv_folds=4 --units_dense=64"
param18="--num_epochs=2 --iter_name=bi_lstm_base_callbackLR --train_embeddings=1 --model_name=bi_lstm --reduction_dim_type=3 --hidden_nodes=64 --cv_folds=4 --units_dense=64 --callbacks_keras=2"
param19="--num_epochs=2 --iter_name=bi_lstm_base_callbackCLR --train_embeddings=1 --model_name=bi_lstm --reduction_dim_type=3 --hidden_nodes=64 --cv_folds=4 --units_dense=64 --callbacks_keras=1"
param20="--num_epochs=2 --iter_name=bi_lstm_base_minLR00001 --train_embeddings=1 --model_name=bi_lstm --reduction_dim_type=3 --hidden_nodes=64 --cv_folds=4 --units_dense=64 --callbacks_keras=2 --min_lr=0.0001"
#---
param21="--num_epochs=2 --iter_name=bi_lstm_base_atten_4fold --train_embeddings=1 --model_name=bi_lstm --reduction_dim_type=1 --hidden_nodes=64 --cv_folds=4 --units_dense=64 --callbacks_keras=2 --min_lr=0.0001"
param22="--num_epochs=3 --iter_name=bi_lstm_base_atten_4fold_3ep --train_embeddings=1 --model_name=bi_lstm --reduction_dim_type=1 --hidden_nodes=64 --cv_folds=4 --units_dense=64 --callbacks_keras=2 --min_lr=0.0001"
param23="--num_epochs=3 --iter_name=bi_lstm_base_atten_4fold_3ep_CLR --train_embeddings=1 --model_name=bi_lstm --reduction_dim_type=1 --hidden_nodes=64 --cv_folds=4 --units_dense=64 --callbacks_keras=1 --min_lr=0.001"

paramslist=("$param21" "$param22" "$param23")

for ((i = 0; i < ${#paramslist[@]}; i++)); do
  echo "$(date): CUDA_VISIBLE_DEVICES=0 python exec_all.py ${paramslist[$i]}"
  CUDA_VISIBLE_DEVICES=0 python exec_all.py ${paramslist[$i]}
done

# sudo shutdown

