from utils import train_data

class series_data_generator(object):
    
    def __init__(self, data, data_columns):
        self.data = data
        self.num_records = data.shape[0]
        self.data_columns = data_columns
        self.index_counter = -1
        self.time_series_column_names = self._build_time_series_column_names()
    
    def get_total_counts(self):
        return self.data.shape[0]

    def _build_time_series_column_names(self):
        column_names = []
        for time_stamp in self.data_columns.step_time_strings:
            column_name = []
            for name in self.data_columns.feature_name_strings:
                column_name.append(name + '_' + time_stamp)
            column_names.append(column_name)
        return column_names


    def _next_index(self):
        self.index_counter += 1
        #if self.index_counter % 1000 == 0:
            #print 'counting on: ', self.index_counter
        if self.index_counter >= self.data.shape[0]:
            self.index_counter = self.index_counter - self.data.shape[0]
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
            one_event = []
            for column_name in self.time_series_column_names:
                one_event.append(self.data.ix[index, column_name].tolist())
            data.append(one_event)
        return data

    def next_batch(self, batch_size):
        index_list = self._batch_index(batch_size)
        ## build the time_series variables
        time_sries_data = self._extract_time_series_data(index_list)
    
        ## build the meta_data and target
        meta_data, target = [], []
        for cur_index in index_list:
            meta_data.append(self.data.ix[cur_index, self.data_columns.meta_data_columns].tolist())
            target.append([self.data.ix[cur_index, self.data_columns.target_column]])
        return train_data(time_series_data = time_sries_data, meta_data = meta_data, target = target)
