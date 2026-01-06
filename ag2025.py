import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from sklearn.neighbors import NearestNeighbors

def get_data(dataset_name, fill_invalid_with=np.nan):

    # Folder and file paths
    folder = r'data/'
    file_name = f"{dataset_name}.npz"
    path = folder + file_name

    # Load data
    data = np.load(path)
    tr_rss = data['tr_rss']
    ts_rss = data['ts_rss']
    tr_crd = data['tr_crd']
    ts_crd = data['ts_crd']
    invalid_value = data['nan_value']
    multi_flr_id = data['multi_fl_id']
    multi_bd_id = data['multi_bd_id']
    fl_type = data['fl_type']

    # Process floor data
    if multi_flr_id == 1:
        fl_ind = -2 if multi_bd_id == 1 else -1

        if fl_type == 'rel':  # Convert floor to regression values if needed
            tr_crd[:, fl_ind] = fl_cls2reg(tr_crd[:, fl_ind], dataset_name)
            ts_crd[:, fl_ind] = fl_cls2reg(ts_crd[:, fl_ind], dataset_name)

        # Extract regression and classification floor features
        tr_floor_cls = fl_reg2cls(tr_crd[:, fl_ind])
        ts_floor_cls = fl_reg2cls(ts_crd[:, fl_ind])
        tr_floor_reg = tr_crd[:, fl_ind]
        ts_floor_reg = ts_crd[:, fl_ind]
    else:
        # No multi-floor information
        tr_floor_cls = np.zeros(tr_crd.shape[0])
        ts_floor_cls = np.zeros(ts_crd.shape[0])
        tr_floor_reg = np.zeros(tr_crd.shape[0])
        ts_floor_reg = np.zeros(ts_crd.shape[0])

    # Replace invalid RSS values
    if fill_invalid_with != "No_Op":
        tr_rss[tr_rss == invalid_value] = fill_invalid_with
        ts_rss[ts_rss == invalid_value] = fill_invalid_with

    # Combine all features into a single array
    if multi_bd_id == 1:
        # Include building data as a column
        tr_building = tr_crd[:, -1]
        ts_building = ts_crd[:, -1]
        tr_combined = np.column_stack((tr_floor_cls, tr_building, tr_crd[:, 0], tr_crd[:, 1], tr_floor_reg))
        ts_combined = np.column_stack((ts_floor_cls, ts_building, ts_crd[:, 0], ts_crd[:, 1], ts_floor_reg))
    else:
        # Exclude building data
        tr_combined = np.column_stack((tr_floor_cls, tr_crd[:, 0], tr_crd[:, 1], tr_floor_reg))
        ts_combined = np.column_stack((ts_floor_cls, ts_crd[:, 0], ts_crd[:, 1], ts_floor_reg))

    return ts_rss, ts_combined, tr_rss, tr_combined

def fl_cls2reg(fl,dataset_name):
    fl_high = {
        'LIB': 2.65,
        'Uji':3.5,
        'TUT3':3.5
    }
    dif = fl_high[dataset_name]
    min_value = np.min(fl)
    scaled_categories = (np.array(fl) - min_value) * dif
    return scaled_categories

def fl_reg2cls(fl):
    unique_values = np.unique(fl)
    value_to_category = {value: index for index, value in enumerate(unique_values)}
    categories = np.array([value_to_category[value] for value in fl])
    return categories

def z_score_normalize(data_row):

    mean = np.mean(data_row)
    std_dev = np.std(data_row)


    if std_dev == 0:
        # Avoid division by zero; return zeros if all values are identical
        return np.zeros_like(data_row)

    return np.asarray([(x - mean) / std_dev for x in data_row])

def l2_normalize(data_row):
    l2_norm = np.linalg.norm(data_row)
    return np.asarray([x / l2_norm for x in data_row])

def rss_normalize(rss_matrix, method='z'):

    if isinstance(rss_matrix, np.ndarray) and rss_matrix.ndim == 1:
        # Convert 1D array to 2D with one row
        rss_matrix = rss_matrix.reshape(1, -1)

    # Select the normalization function
    if method == 'z':
        normalize_func = z_score_normalize
    elif method == 'l2':
        normalize_func = l2_normalize
    else:
        raise ValueError("Invalid method. Use 'z' for Z-score or 'l2' for L2 normalization.")

    # Normalize each row
    normalized_matrix = np.array([normalize_func(row) for row in rss_matrix])

    return normalized_matrix

def getfp(train_x, train_y, norm_X, test_point, cos_p=3):

    # Step 1: Compute Cosine Distance for all training points
    distances = cosine_distances(test_point.reshape(1, -1), norm_X).flatten()

    # Find the lowest and largest distance
    lowest_dis = np.min(distances)
    largest_dis = np.max(distances)

    # Define the interval for distances
    lower_bound = lowest_dis
    upper_bound = lowest_dis + (largest_dis - lowest_dis) * cos_p / 100

    # Step 2: Find indices within the defined interval
    index_1 = np.where((distances >= lower_bound) & (distances <= upper_bound))[0]

    # Filter the training data based on the selected indices
    abstracted_train_x = train_x[index_1, :]
    abstracted_train_y = train_y[index_1, :]

    return abstracted_train_x, abstracted_train_y

class KNNRegression:
    def __init__(self, k=3, test_point=None):
        self.k = k
        self.test_point = test_point

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def distance(self, x1, x2):
        return np.linalg.norm(x1 - x2)

    def predict(self, X):
        if self.test_point is not None:
            return np.array(self._predict_single(X[self.test_point,:]))
        predictions = [self._predict_single(x) for x in X]
        return np.array(predictions)

    def _predict_single(self, x):
        # Compute distances to all training points
        distances = [self.distance(x, x_train) for x_train in self.X_train]

        if self.test_point is not None:
            return np.array(distances)\

        # Get indices of the k-nearest neighbors
        k_indices = np.argsort(distances)[:self.k]

        # Extract coordinates of the k-nearest neighbors
        k_targets = [self.y_train[i] for i in k_indices]

        # Return the average of the k-nearest coordinates
        return np.mean(k_targets, axis=0)

def get_stat(data1,data2):
    errors = euclidean_distance(data1, data2)
    sorted_errors = np.sort(errors)
    mae = np.mean(errors)
    p80 = np.percentile(sorted_errors, 80)
    pb10 = (np.sum(errors < 10) / len(errors)) * 100
    return mae, p80, pb10

def euclidean_distance(data1,data2):
    distances = np.linalg.norm(data1 - data2, axis=1)
    return distances

def get_k_nearest(input_data, X_train, Y_train, k=8, nan_replace_value=-100):
    # Adjust k if necessary (if k is greater than the number of samples in X_train)
    num_samples = X_train.shape[0]
    if k > num_samples:
        k = num_samples  # Set k to the number of available samples
        # print('k change to',k)

    # Initialize and fit the NearestNeighbors model
    knn_model = NearestNeighbors(n_neighbors=k, algorithm='auto')
    knn_model.fit(X_train)

    # Reshape the input data if necessary
    input_data = input_data.reshape(1, -1)

    # Find the k nearest neighbors
    distances, indices = knn_model.kneighbors(input_data)

    # Retrieve the k nearest neighbors from X_train and their corresponding Y_train values
    k_nearest_X = X_train[indices[0]]
    k_nearest_Y = Y_train[indices[0]]

    return k_nearest_X, k_nearest_Y

