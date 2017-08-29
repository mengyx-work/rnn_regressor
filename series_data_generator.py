from utils import train_data, full_column_name_by_time


class SeriesDataGenerator(object):
    """ class used to generate training/test data in batch

    this class maintains a position index counter as ``index_counter`` to keep track of
    the next_index

    only external-exposed function is `next_batch`; given the ``batch_size``, this function
    will collect the data index by index, and return data in the format of namedTuple `train_data`.


    Attributes:
        data (pd.DataFrame): the training/test data
        config_dict (Dict): dict contains different lists of columns names
        time_series_column_names (list of list): the time series columns in sequence, created by
        function `_build_time_series_column_names`
    """

    def __init__(self, data, config_dict):
        self.config_dict = config_dict.copy()
        self.index_counter = -1
        self.total_row_counts = self._get_total_counts(data)
        self.time_series_column_names = self._build_time_series_column_names()
        self.instance_sequence = self._prepare_instance_sequence(data)
        self._cur_index = 0

    def _prepare_instance_sequence(self, data):
        data = data.sample(frac=1)
        instance_sequence = []
        for index in xrange(self.total_row_counts):
            instance = {}
            instance_time_series = []
            for column_name_set in self.time_series_column_names:
                instance_time_series.append(data.iloc[index][column_name_set].tolist())
            instance['time_series_data'] = instance_time_series
            instance['meta_data'] = data.iloc[index][self.config_dict["static_columns"]].tolist()
            instance['target'] = [data.iloc[index][self.config_dict["label_column"]]]
            instance_sequence.append(instance)
        return instance_sequence

    @staticmethod
    def _get_total_counts(data):
        return data.shape[0]

    def _build_time_series_column_names(self):
        column_names = []
        for time_stamp in self.config_dict['time_step_list']:
            single_step_column_name = []
            for name in self.config_dict["time_interval_columns"]:
                single_step_column_name.append(full_column_name_by_time(name, time_stamp))
            column_names.append(single_step_column_name)
        return column_names

    @staticmethod
    def _add_instances_by_index(instance_sequence, start_index, end_index, time_series_data_array, meta_data_array, target_array):
        for index in xrange(start_index, end_index):
            time_series_data_array.append(instance_sequence[index]['time_series_data'])
            meta_data_array.append(instance_sequence[index]['meta_data'])
            target_array.append(instance_sequence[index]['target'])

    def next_batch(self, batch_size):
        time_series_data, meta_data, target = [], [], []
        if self._cur_index + batch_size <= self.total_row_counts:
            start_index = self._cur_index
            self._cur_index += batch_size
            self._add_instances_by_index(self.instance_sequence, start_index, self._cur_index, time_series_data, meta_data, target)
        else:
            start_index = self._cur_index
            self._cur_index = self._cur_index + batch_size - self.total_row_counts
            self._add_instances_by_index(self.instance_sequence, start_index, self.total_row_counts, time_series_data, meta_data, target)
            self._add_instances_by_index(self.instance_sequence, 0, self._cur_index, time_series_data, meta_data, target)
        return train_data(time_series_data=time_series_data, meta_data=meta_data, target=target)