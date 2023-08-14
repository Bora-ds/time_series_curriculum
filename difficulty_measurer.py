import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# 일단 모든 항목에 대해서 0~1 스케일링 후 다 더하고 평균으로 줄세우기.
# 단, 음수값은 모두 양의 값으로 변환 후 스케일링!! (음의 값인 경우 작으면 그 정도가 심한 경우라면)

# diversity
'''
Trend
Seasonality
Skewness
Kurtosis
Periodicity
'''

def score_diversity(x_train):

    # skewness***********************

    skewness_list = []
    kurtosis_list = []

    for a in x_train:
        # x_train에서 각 record 별로 iterate
        skewness_list.append(abs(skew(a)[0].astype(float))) # 절댓값 적용함
        kurtosis_list.append(kurtosis(a, fisher=True))

    print('loook at skewness')
    #print(skewness)

    # 0~1로 스케일링
    #skewness_scaler=StandardScaler()
    minmaxcaler = MinMaxScaler()
    skewness_scaled = minmaxcaler.fit_transform(np.reshape(skewness_list, (-1, 1)))
    kurtosis_scaled = minmaxcaler.fit_transform(np.reshape(kurtosis_list, (-1, 1)))
    diversity = skewness_scaled + kurtosis_scaled
    #complexity = np.empty(x_train.shape[0])
    #print(skewness_scaled)


    #Kurtosis***********************
    # The distribution with a "higher" kurtosis has a "heavier" tail. -> 값이 작을 수록 sparse 하다고 봐야 함. -> 0~1로 스케일링한 후, 1에서 빼야 함.
    # The zero valued kurtosis of the normal distribution in Fisher’s definition can serve as a reference point.
    to_draw = np.isfinite(diversity)
    plt.hist(to_draw, bins=50)
    plt.gca().set(title='Kurtosis Histogram', ylabel='Frequency')
    # plt.show()
    # plt.close()
    plt.savefig('D:\\OneDrive\\대학원\\연구\\실험\\kurtosis_histogram.png')
    plt.clf()




    return diversity

# complexticy
'''
Non-linearity
Self-similarity
- peak의 개수
- smoothed 했을 때의 그 차이의 합 또는 그 차이가 0.x 이상인 것의 개수
Length
'''

def score_complexity(x_train):

    # input은 데이터
    # 순위별로???
    #print(x_train)

    # length
    length_list = []
    num_peaks_list = []

    # peak 의 기준:
    # 전체 y 최대 - 최소의 0.3 이상?
    # 가장 많이 쓰는 기준은??
    # 실제 데이터를 그려보고
    for a in x_train:
        # x_train에서 각 record 별로 iterate
        a = a.flatten()
        #print(a)
        #print(type(a))
        #print(a.shape)

        length_list.append(len(a))  # 절댓값 적용함
        #peak_threshold = (max(a)-min(a))*0.1
        peaks, _ = find_peaks(a)
        num_peaks = len(peaks)
        num_peaks_list.append(num_peaks)

    print('num_peaks_list')
    print(num_peaks_list)
    minmaxcaler = MinMaxScaler()
    length_scaled = minmaxcaler.fit_transform(np.reshape(length_list, (-1, 1)))
    num_peaks_scaled = minmaxcaler.fit_transform(np.reshape(num_peaks_list, (-1, 1)))

    complexity = length_scaled + num_peaks_scaled
    print(x_train.shape[0])
    # complexity = np.empty(x_train.shape[0])

    return complexity



def score_noise(x_train, y_train):

    # input은 데이터
    # 순위별로???
    #print(x_train)

    # length

    print('complexity shape : ')
    print(x_train.shape[0])
    #complexity = np.empty(x_train.shape[0])



# noise
'''
chaotic
class imbalance
Serial correlation
'''
