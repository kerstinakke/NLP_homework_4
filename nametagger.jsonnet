local embedding_dim = 58;
local hidden_dim = 128;
local num_epochs = 1000;
local patience = 10;
local batch_size = 64;
local learning_rate = 0.005;

{
    "train_data_path": "data\\names",
    "dataset_reader": {
        "type": "homework4-reader"
    },
    "model": {
        "type": "homework4-model",
        "encoder": {
            "type": "rnn",
            "input_size": embedding_dim,
            "hidden_size": hidden_dim
        }
    },
    "iterator": {
        "type": "bucket",
        "batch_size": batch_size,
        "sorting_keys": [["name", "num_tokens"]]
    },
    "trainer": {
        "num_epochs": num_epochs,
        "optimizer": {
            "type": "sgd",
            "lr": learning_rate
        },
        "patience": patience
    }
}
