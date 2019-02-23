import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--model_name',
                    type = str,
                    default = 'bi_lstm',
                    help = 'name of the model')
parser.add_argument('--iter_name',
                    type = str,
                    default = 'iter0',
                    help = 'iteration name')
parser.add_argument('--max_lr',
                    type = float,
                    default = 0.002,
                    help = 'maximum learning rate value')
parser.add_argument('--min_lr',
                    type = float,
                    default = 0.001,
                    help = 'minimum learning rate value')
parser.add_argument('--optimizer',
                    type = str,
                    default = 'adam',
                    help = 'Optimizer for the NN')
parser.add_argument('--units_dense',
                    type = int,
                    default = 16,
                    help = 'Number of units in the final layer of the N.N')
parser.add_argument('--hidden_nodes',
                    type = int,
                    default = 40,
                    help = 'Number of units in the LSTM / GRU layers ')
parser.add_argument('--num_epochs',
                    type = int,
                    default = 3,
                    help = 'Number of epochs for training')
parser.add_argument('--reduction_dim_type',
                    type = int,
                    default = 1,
                    help = 'Attention or Max/Avg Pooling or both')
parser.add_argument('--dropout_final_layer',
                    type = float,
                    default = 0.1,
                    help = 'Dropout in the final layer')
parser.add_argument('--train_embeddings',
                    type = int,
                    default = 0,
                    help = 'whether or not to re-train the embeddings')
parser.add_argument('--add_new_embedding',
                    type = int,
                    default = 0,
                    help = 'whether or not to use a new embedding')
parser.add_argument('--maxlen',
                    type = int,
                    default = 100,
                    help = 'whether or not to use a new embedding')
parser.add_argument('--print_log',
                    type = int,
                    default = 1,
                    help = 'whether or not to print onto a log file')
parser.add_argument('--use_wiki',
                    type = int,
                    default = 0,
                    help = 'whether or not to include wikipedia pretrained embedding')
parser.add_argument('--callbacks_keras',
                    type = int,
                    default = 0,
                    help = 'whether or not to use dynamic Learning Rate')
parser.add_argument('--batch_size',
                    type = int,
                    default = 512,
                    help = 'Batch size')
parser.add_argument('--vocab_size',
                    type = int,
                    default = 95000,
                    help = 'Batch size')
parser.add_argument('--cv_folds',
                    type = int,
                    default = 5,
                    help = 'Number of CV folds')