import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import nltk
import string

# Загрузка stopwords
nltk.download('stopwords')

# Загрузка данных
data = pd.read_csv('C:/Users/Федот/PycharmProjects/Modelforsearchpractice/files/captions.txt')

# Очистка текста
stop_words = set(stopwords.words('english'))
def clean_text(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    text = ' '.join([word if word not in stop_words else '<UNK>' for word in text.split()])
    return text

data['caption'] = data['caption'].apply(clean_text)

# Анализ длины описаний
lengths = data['caption'].apply(lambda x: len(x.split()))
plt.hist(lengths, bins=20, color='blue', edgecolor='black')
plt.title('Распределение длины текстовых описаний')
plt.xlabel('Количество слов')
plt.ylabel('Частота')
plt.grid(True)
plt.savefig('caption_lengths.png')
plt.close()