# coding=utf-8
import pandas as pd
import numpy as np
#  Импортирование исходного датасета, присвоение названия каждой колонке.
#df = pd.read_csv('https://storage.googleapis.com/semantic-duplicates-fc2cb7640350/yandex_rss_2016_.csv')
#df.columns = ['uuid', 'short', 'long', 'label', 'date', 'source']
def form_dataset(f1, f2):
    df = pd.DataFrame(columns=['both'], index=[0, 1])
    result = pd.DataFrame(columns=['index_0', 'index_1'], index=[0])
    df['both'][0] = f1
    df['both'][1] = f2
    result['index_0'] = 0
    result['index_1'] = 1	
    return df, result

# Создание функции для обучения классификатора. На вход поступают колонка со значениями True/False а также датафрейм,
# содержащий колонки с признаками
def classifier(df_selected):
    # преобразовываем df_selected в словарь
    df_features = df_selected.to_dict(orient='records')
    # импортируем DictVectorizer, проводим векторизацию признаков с сохранением в features
    from sklearn.feature_extraction import DictVectorizer
    vec = DictVectorizer()
    features = vec.fit_transform(df_features).toarray()
    from sklearn.externals import joblib
    clf = joblib.load('trained_model.pkl')
    answers = clf.predict(features)
    predictions = clf.predict_proba(features)
    return answers, predictions

import nltk
from nltk.corpus import stopwords
from pymystem3 import Mystem
mystem = Mystem(entire_input=False)
import re
# С учетом особенностей модуля mystem (который удаляет слова, содержащие цифры, из текста), создадим функцию,
# которая будет предварительно убирать цифры, стоп-слова и слова,
# содержащие две и менее букв, а затем применим анализатор от mystem, который предоставит начальную форму слов и
# грамматически охарактеризует их
def mystem_combined(file_text):
    # декодируем текст
    #file_text = file_text.decode('utf-8')
    # токенизация текста при помощи nltk.word_tokenize (модуль nltk)
    tokens = nltk.word_tokenize(file_text)
    # все буквы в словах заменяем на строчные
    tokens = [token.lower() for token in tokens]
    # удаляем цифры
    tokens = [re.sub(r'\d', '', i) for i in tokens]
    # удаляем стоп-слова
    stop_words = stopwords.words('russian')
    tokens = [i for i in tokens if i not in stop_words]
    # удаляем слова, содержащие две или менее букв
    tokens = [i for i in tokens if not len(i) <= 2]
    # сливаем образовавшиеся токены в один текст
    file_text = ' '.join(tokens)
    # применяем анализатор mystem
    file_text = mystem.analyze(file_text)
    # теперь для каждого слова хранится грамматическая информация, начальная форма и исходный текст. Удалим исходный текст
    # за ненадобностью и в целях уменьшения объема памяти, занимаемого информацией о словах. Так как mystem применим
    # только к русскоязычным словам, то для английских слов и нераспознанных русских слов грамматической информации не
    # имеется. Поэтому при обращении к ней может возникнуть ошибка IndexError. Будем пропускать такие слова.
    for k in range(len(file_text)):
        try:
            file_text[k][u'analysis'][0][u'gr']
            del file_text[k][u'text']
        except IndexError:
            pass
    # создаем новый список, добавляем к нему словари с информацией о словах. Если словари полностью совпадают, то лишний
    # словарь не добавляется в список
    new_file_text = []
    for x in file_text:
        if x not in new_file_text:
            new_file_text.append(x)
    # возвращаем список
    return new_file_text

#df['both'] = df.short + ' ' + df.long
#mystem_combined(df.both)


# создадим функцию, которая приведет всю информацию о слове к виду: word, x, y, где word - начальная форма слова,
# x - код части речи, y - цифра "0" - слово не является географическим названием, не является именем, фамилией или отчеством,
# цифра "1" - слово является географическим названием, цирфа "2" - слово является именем, фамилией или отчеством
def obrabotka(series):
    # имеется предварительно созданный файл, в котором каждой части речи присвоено какое-либо значение от 0 до 14, где
    # цифра 0 обозначает, что часть речи не определена. Это может говорить о том, что слово не русскоязычное либо это слово
    # не определено словарем (например, из-за ошибок в слове)
    # импортируем файл characterisics.csv и дадим название его колонкам - word class и code
    table_ch = pd.read_csv('characteristics.csv', sep=',', header=None)
    table_ch.columns = ['word class', 'code']

    # импортируем модуль string (необходим для нахождения знаков препинания)
    import string
    # создадим предварительно вспомогательную функцию geo_name, которая будет часто использоваться в основной функции obrabotka
    # на вход поступают слово m и список l
    def geo_name(m, l):
        # если слово является географическим названием, к списку l добавить цифру "1"
        if m == u'гео':
            l.append(1)
        # если слово является именем, фамилией или отчеством, к списку l добавить цифру "2"
        elif m == u"имя" or m == u"фам" or m == u"отч":
            l.append(2)
        # если слово не имеет этих признаков, к списку l добавить цифру "0"
        else:
            l.append(0)
        # возвращаем l
        return l
    # обращаемся к каждой строке столбца
    for p in range(series.shape[0]):
        # присваиваем строку line
        line = series[p]
        # обращаемся к каждому словарю строки, в котором содержится информация о слове
        for y in range(len(line)):
            # вводим переменные: 1) n - необходим для счета количества обработанных слов в строчке, в которой дается
            # грамматическая характеристика слова; это число не должно превышать двух, когда n = 1, это значит,
            # что определена часть речи, когда n = 2, это значит, что определена часть речи и признак "гео", "имя" и т.п.
            # 2) l - список, который и будет содержать новую информацию о слове; 3) m - слова, здесь происходит "сборка"
            # слова по буквам
            n = 0
            l = []
            m = ''
            # пробуем присоединить к списку l начальную форму слова
            try:
                l.append(line[y][u'analysis'][0][u'lex'])
                # проверяем, не было ли проверено уже два слова из раздела грамматики, если ответ утвердительный, то
                # исходная строка заменяется списком l.
                for k in range(len(line[y][u'analysis'][0][u'gr'])):
                    if n == 2:
                        pass
                    # в противном случае проверяем, является ли отдельно взятая буква из раздела грамматики знаком препинания
                    # (используется модуль string)
                    else:
                        if line[y][u'analysis'][0][u'gr'][k] in string.punctuation:
                            # если это так, то проверяется, является ли следующий символ знаком препинания. Если ответ
                            # утвердительный, то ничего не делается (операция необходима для случаев, когда идут подряд
                            # два знака препинания. Если пропускать этот шаг, то второй слово из грамматического раздела
                            # может быть пропущено, а это значит, что возможно упускается признак)
                            # при этом может возникнуть ошибка - IndexError.
                            try:
                                if line[y][u'analysis'][0][u'gr'][k+1] in string.punctuation:
                                    pass
                                else:
                                    # если следующий символ не является знаком препинания, то проводим следующие операции
                                    # если n = 0, то нужно выделить часть речи
                                    if n == 0:
                                        l.append(table_ch[table_ch['word class'] == m].index[0])
                                    # если n = 1, то нужно попробовать выделить географический или именной признак.
                                    # Для этого обращаемся к вспомогательной функции.
                                    if n == 1:
                                        l = geo_name(m, l)
                                    # Значение m обнуляем, чтобы затем наполнить его буквами следующего слова
                                    m = ''
                                    # значение n увеличиваем на единицу
                                    n += 1
                            except IndexError:
                                # если такая ошибка возникла, то нужно проверить значение n. Если оно равно нулю, то
                                # к списку l добавляется код части речи слова. Так как больше информации о слове нет,
                                # то и географических и именных признаков нет, поэтому к списку также добавляется число "0".
                                if n == 0:
                                    l.append(table_ch[table_ch['word class'] == m].index[0])
                                    l.append(0)
                                # если значение n равно единице, то это значит, что часть речи была выделена, и нужно
                                # проверить, является ли второе слово географическим или именным признаком. Для этого
                                # обращаемся к вспомогательной функции geo_name.
                                if n == 1:
                                    l = geo_name(m, l)
                                # больше информации о слове нет, поэтому приравниваем n к двум.
                                n == 2
                        else:
                            # если следующий символ не явлется знаком препинания, то "собирание" слова еще не завершено.
                            # Поэтому к m присоединяем символ.
                            m += line[y][u'analysis'][0][u'gr'][k]
                            try:
                                # проверяем, существует ли следующий символ
                                line[y][u'analysis'][0][u'gr'][k+1]
                            except IndexError:
                                # если его не существует, то это значит, что грамматическая информация о слове закончилась.
                                # Так как в ней содержится по крайней мере одно слово, то n в данном случае не может быть
                                # равен нулю. Значит, часть речи уже выделена, остается попытаться выделить географический
                                # или именной признак. Обращаемся к вспомогательной функции geo_name.
                                if n == 1:
                                    l = geo_name(m, l)
                                    n = 2
                # После всех операций над символами заменяем исходную строку на список l.
                line[y] = l
            # если при попытке присоединения к списку l начальной формы слова возникает ошибка IndexError, то
            # в список l добавляется исходная форма слова, считается, что никаких признаков слово не имеет
            except IndexError:
                l.append(line[y][u'text'])
                l.append(0)
                l.append(0)
            # исходную информацию о слове заменяем информацией из списка l
                line[y] = l


# После выделения именных и географических признаков переходим к выделению количественных признаков.
def features(df, result):
	# создаем функцию, при выполнении которой получим датафрейм с количеством совпадающих слов 		двух новостей
    	# по той или иной части речи. На вход поступает колонка датафрейма.

        # создаем columns, которому присваиваем значения от 0 до 14
        columns = np.arange(15)
        # создаем три датафрема с колонками из columns, количество строк определено количеством 	строк датафрема result
        df_class1 = pd.DataFrame(columns=columns, index=np.arange(result.shape[0]))
        df_class2 = pd.DataFrame(columns=columns, index=np.arange(result.shape[0]))
        df_class3 = pd.DataFrame(columns=columns, index=np.arange(result.shape[0]))
        for k in range(len(result)):
            p = result.index_0[k]
            for m in range(len(df.both[p])):
                try:
                    df_class1[df.both[p][m][1]][k].add(df.both[p][m][0])
                except AttributeError:
                    df_class1[df.both[p][m][1]][k] = set()
                    df_class1[df.both[p][m][1]][k].add(df.both[p][m][0])
            p1 = result.index_1[k]
            for m in range(len(df.both[p1])):
                try:
                    df_class2[df.both[p1][m][1]][k].add(df.both[p1][m][0])
                except AttributeError:
                    df_class2[df.both[p1][m][1]][k] = set()
                    df_class2[df.both[p1][m][1]][k].add(df.both[p1][m][0])
            for l in range(15):
                try:
                    df_class3[l][k] = len(df_class1[l][k].intersection(df_class2[l][k]))
                except AttributeError:
                    df_class3[l][k] = 0
                except TypeError:
                    df_class3[l][k] = 0
    	return df_class3


#classifier(geo_class.same, geo_class.drop(['same'], axis=1))
