class gl:
    pkl_df_photo_urls = "pkl_df_photo_urls.pkl"
    pkl_model_conv = "model_conv.pkl"
    pkl_train_loader = "pkle_train_loader.pkl"
    pkl_val_loader = "pkle_val_loader.pkl"
    pkl_test_loader = "pkle_test_loader.pkl"
    # column names: ['tweetID', 'crDate','edInput', 'editor', 'engages', 'isApproved', 'isEdNeed', 'isRT', 'likes', 'photoUrl', 'retweets', 'rtUsID', 'text', 'topicName', 'usFlwrs', 'usID', 'usName', 'videoUrl']
    photoUrl = "photoUrl"
    topic = "topicName"
    business = "Business"
    image_path = "ImagePath"
    is_business = "IsBusiness"
    usID = "usID"
    usName = "usName"
    output_folders = ["Pickle"]

def get_perc(sample, population):
    return round((sample/population * 100), 2)

def get_dictionaries_from_list(list):
    dict_rev=dict(enumerate(list))
    new_dict = dict([(value, key) for key, value in dict_rev.items()])
    return new_dict, dict_rev