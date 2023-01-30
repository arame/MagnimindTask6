class gl:
    pkl_df_photo_urls = "pkl_df_photo_urls.pkl"
    # column names: ['tweetID', 'crDate','edInput', 'editor', 'engages', 'isApproved', 'isEdNeed', 'isRT', 'likes', 'photoUrl', 'retweets', 'rtUsID', 'text', 'topicName', 'usFlwrs', 'usID', 'usName', 'videoUrl']
    photoUrl = "photoUrl"
    topic = "topicName"
    nature = "Nature"
    is_nature = "IsNature"
    usID = "usID"
    usName = "usName"
    output_folders = ["Pickle"]

def get_perc(sample, population):
    return round((sample/population * 100), 2)