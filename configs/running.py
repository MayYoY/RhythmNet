class TrainConfig:
    trans = None
    record = "./record.csv"
    folds = [2, 3, 4, 5]
    tasks = []
    sources = [1, 2, 3]

    batch_size = 16
    num_epochs = 60
    mask = False

    device = "cuda:1"  # cuda:1
    device_ids = [4, 5, 6, 7]


class TestConfig:
    trans = None
    record = "./record.csv"
    folds = [1]
    tasks = []
    sources = [1, 2, 3]

    batch_size = 1
    mask = False
    methods = ["Mean", "Std", "MAE", "RMSE", "MAPE", "R"]

    device = "cuda:1"  # cpu
