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
        self.data = data
        self.config_dict = config_dict.copy()
        self.index_counter = -1
        self.total_row_counts = self.get_total_counts()
        self.time_series_column_names = self._build_time_series_column_names()

    def get_total_counts(self):
        return self.data.shape[0]

    def _build_time_series_column_names(self):
        column_names = []
        for time_stamp in self.config_dict['time_step_list']:
            single_step_column_name = []
            for name in self.config_dict["time_interval_columns"]:
                single_step_column_name.append(full_column_name_by_time(name, time_stamp))
            column_names.append(single_step_column_name)
        return column_names

    def _next_index(self):
        self.index_counter += 1
        if self.index_counter >= self.total_row_counts:
            self.index_counter = self.index_counter - self.total_row_counts
            return self.index_counter 
        else:
            return self.index_counter

    def _batch_index(self, batch_size):
        index_list = []
        for _ in range(batch_size):
            index_list.append(self._next_index())
        return index_list

    def _extract_time_series_data(self, index_list):
        data = []
        for index in index_list: 
            instance = []
            for column_name_set in self.time_series_column_names:
                instance.append(self.data.iloc[index][column_name_set].tolist())
            data.append(instance)
        return data

    def next_batch(self, batch_size):
        index_list = self._batch_index(batch_size)
        # build the time_series variables
        time_series_data = self._extract_time_series_data(index_list)
        # build the meta_data and target
        meta_data, target = [], []
        for cur_index in index_list:
            meta_data.append(self.data.iloc[cur_index][self.config_dict["static_columns"]].tolist())
            target.append([self.data.iloc[cur_index][self.config_dict["label_column"]]])
        return train_data(time_series_data=time_series_data, meta_data=meta_data, target=target)
