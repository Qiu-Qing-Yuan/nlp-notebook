from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import pandas as pd

df = ["He is a good dog.", "The dog is too lazy.",
         "That is a brown cat.", "The cat is very active.", "I have brown cat and dog."]

tv = TfidfVectorizer(stop_words='english', smooth_idf=True)

tv_fit = tv.fit_transform(df)
# 原始单词文本矩阵
print(tv.get_feature_names_out())
print(tv_fit.toarray())


# SVD represent documents and terms in vectors
svd_model = TruncatedSVD(n_components=2, algorithm='randomized', n_iter=100)
lsa = svd_model.fit_transform(tv_fit)

dictionary = tv.get_feature_names_out()
encoding_matrix = pd.DataFrame(svd_model.components_, index = ["topic_1","topic_2"], columns = (dictionary)).T
clean_ = tv.get_feature_names_out()
list = list.append(clean_)
print(list)
