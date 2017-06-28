import os, collections

label_col = 'total_views'
index_col = 'articleId'
#dep_var_col = 'total_views'

## the namedtuple for data collection
train_data = collections.namedtuple('train_data', ['time_series_data', 'meta_data', 'target'])
data_columns = collections.namedtuple('data_columns', ['step_time_strings', 'feature_name_strings', 'meta_data_columns', 'target_column'])

## a namedtuple of feature names
independent_features = ['articleInfo_type', 'articleInfo_authorName', 'articleInfo_section', 'minLocalDateInWeek', 'minLocalTime', 'createTime', 'publishTime']
feature_name_strings = ['views_PageView', 'sessionReferrer_DIRECT_PageView', 'pageReferrer_OTHER_PageView', 'platform_PHON_PageView', 'platform_DESK_PageView', 'sessionReferrer_SEARCH_PageView', 'pageReferrer_SEARCH_PageView', 'platform_TBLT_PageView', 'pageReferrer_DIRECT_PageView', 'sessionReferrer_SOCIAL_PageView', 'pageReferrer_EMPTY_DOMAIN_PageView', 'pageReferrer_SOCIAL_PageView']
step_time_strings = ['0min_to_10min', '10min_to_20min', '20min_to_30min', '30min_to_40min', '40min_to_50min', '50min_to_60min']

columns = data_columns(step_time_strings = step_time_strings, 
                       feature_name_strings = feature_name_strings, 
                       meta_data_columns = independent_features, 
                       target_column = label_col)


def clear_folder(absolute_folder_path):
    if not os.path.exists(absolute_folder_path):
        os.makedirs(absolute_folder_path)
    for file_name in os.listdir(absolute_folder_path):
        file_path = os.path.join(absolute_folder_path, file_name)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print 'failed to clear folder {}, with error {}'.foramt(absolute_folder_path, e)
