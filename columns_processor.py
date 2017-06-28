class categorical_columns_processor(object):

    def __init__(self, train, valid_data, dep_var_col, level_limit=50, missing_filling_string='missing', column=None):
        self.train = train
        self.valid_data = valid_data
        self.dep_var_col = dep_var_col
        self.level_limit = level_limit
        self.missing_filling_string = missing_filling_string
        if column is not None:
            self.column = column

        self.other_level_string = 'other_level'

    def _create_map_by_level_limit(self, col, level_limit):
         ## build the frequency count
    	counts = self.train[col].value_counts()
    	counts.sort_values(ascending=False)
    	level_limit = level_limit if level_limit < counts.shape[0] else counts.shape[0]
        counts = counts.iloc[:level_limit]
        freq_map = {}
        for level in self.train[col].unique():
            if level in counts.index.tolist():
                freq_map[level] = level
            else:
                freq_map[level] = self.other_level_string

        ## check the unique levels from `valid_data`
        for level in self.valid_data[col].unique():
            if level not in freq_map:
                freq_map[level] = self.other_level_string
        return freq_map

    def _create_encoding_map_from_mean_dep_var(self, tmp_train, freq_map, col):
        ## create mean dep_var for each level
        mean_dep_var = tmp_train.groupby(col).mean()
        encoding_map = {}
        for level in self.train[col].unique():
            new_index = freq_map[level]
            encoding_map[level] = mean_dep_var.loc[new_index, self.dep_var_col]
    
        for level in self.valid_data[col].unique():
            new_index = freq_map[level]
            ## if the level in valid_data not found in train
            ## use the mean dep_var
            if new_index not in mean_dep_var.index:
                encoding_map[level] = mean_dep_var[self.dep_var_col].mean()
            else:
                encoding_map[level] = mean_dep_var.loc[new_index, self.dep_var_col]
        return encoding_map

    def encode_categorical_column(self, column=None, level_limit=None, missing_filling_string=None):
        if column is not None:
            self.column = column
        if level_limit is not None:
            self.level_limit = level_limit
        if missing_filling_string is not None:
            self.missing_filling_string = missing_filling_string

        encoding_map = self._create_column_encoding_map(fill_missing=True)
        return self.train[self.column].copy().fillna(self.missing_filling_string).map(encoding_map), self.valid_data[self.column].copy().fillna(self.missing_filling_string).map(encoding_map)

    def _create_column_encoding_map(self, fill_missing=True):
        if fill_missing:
            self.train = self.train.fillna(self.missing_filling_string)
            self.valid_data = self.valid_data.fillna(self.missing_filling_string)
        ## create frequency_map from both train and valid_data
        freq_map = self._create_map_by_level_limit(self.column, self.level_limit)
        ## a subset of train to map old levels into meta levels
        ## from freq_map
	tmp_train = self.train[[self.column, self.dep_var_col]].copy()
        tmp_train[self.column] = tmp_train[self.column].map(freq_map)
        ## build the final encoding map from the fred_map, a cut-off map
        ## and use mean_dep_var to encode levels
        encode_map = self._create_encoding_map_from_mean_dep_var(tmp_train, freq_map, self.column)
        return encode_map
    

