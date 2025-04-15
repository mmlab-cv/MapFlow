from absl import flags

FLAGS = flags.FLAGS

# misc
flags.DEFINE_integer('seed', 42, 'Random seed used to reproducibility')
flags.DEFINE_bool('wandb', False, 'Enable wandb')
flags.DEFINE_string('dataset_folder', 'dataset', 'Dataset path')
flags.DEFINE_string('results_folder', 'results', 'Destination path')
flags.DEFINE_bool('save_checkpoints', True, 'Enable saving weights')
flags.DEFINE_bool('load_checkpoints', False, 'Enable loading weights')

# input params
flags.DEFINE_bool('pos_emb_enabled', True, 'Enable positional embedding')
flags.DEFINE_integer('conditioning_peds', 3, 'Number of pedestrian in the scene to conditioning the current pedestrian')
flags.DEFINE_integer('min_radius', 0, 'Number of pedestrian in the scene to conditioning the current pedestrian')
flags.DEFINE_bool('closest', True, 'Take closest pedestrian or random')

flags.DEFINE_integer('observation_len', 8, 'Step observed')
flags.DEFINE_integer('prediction_len', 12, 'Step to be predicted')
flags.DEFINE_integer('skip', 1, 'Frame skipped between two video sequences observed')

flags.DEFINE_integer('num_traj', 20, 'Number of trajectories to be sampled during inference')

# network params
flags.DEFINE_integer('B', 15, 'Value of the spline boundaries')
flags.DEFINE_integer('encoding_size', 16, 'Trajectories encoding size')
flags.DEFINE_integer('num_layers', 10, 'Number of flow layers')
flags.DEFINE_integer('K', 8, 'Number of knot points')
flags.DEFINE_integer('hidden_dim', 32, 'Hidden dimension')
flags.DEFINE_bool('rel_pos', True, 'Use relative positions')
flags.DEFINE_bool('norm_rotation', True, 'Normalizing and rotate')

# autoencoder params
flags.DEFINE_bool('ae_enabled', True, 'Enable the autoencoder')
flags.DEFINE_integer('ae_embedding', 16, 'Embedding size')
flags.DEFINE_integer('ae_layers', 3, 'Number of layers of the gru of the autoencoder')
flags.DEFINE_integer('ae_hidden', 16, 'Dim of the hidden layer')
flags.DEFINE_integer('ae_enc', 16, 'Encoding size, currently equal to ae_hidden')

# ETH/UCY
flags.DEFINE_bool('augmentation', True, 'Augmenting data in dataloader')
flags.DEFINE_integer('alpha', 10, 'Value of alpha')
flags.DEFINE_float('beta', 0.2, 'Value of beta')
flags.DEFINE_float('gamma', 0.02, 'Value of gamma')

# training params
flags.DEFINE_bool('train', True, 'Are we training?')

flags.DEFINE_string('device', 'cuda:0', 'Device used')
flags.DEFINE_float('lr', 1e-3, 'Learning rate')
flags.DEFINE_integer('epochs', 150, 'Number of epochs')
flags.DEFINE_integer('validation_rate', 10, 'Validate every n epochs')
flags.DEFINE_integer('print_traj', 10, 'Print sampled trajectories every n epochs')
flags.DEFINE_integer('batch', 128, 'Batch size')
flags.DEFINE_float('tt_split_value', 0.9, 'Train/Test split the dataset (it indicates the percentage of the test set)')
flags.DEFINE_float('tv_split_value', 0.9, 'Train/Val split the dataset (it indicates the percentage of the val set)')

# {eth_uni, hotel, ucy_uni, zara1, zara2}.txt
flags.DEFINE_multi_string('loo_file', ["eth_uni.txt"], 'Leave-one-out approach: file excluded from training')



