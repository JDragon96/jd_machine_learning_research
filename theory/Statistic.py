import numpy as np

def getCovMatrix(data, feature_num):
    """
    .data : 데이터
    .feature_num : 특징벡터 길이
    .return : (feature_num, feature_num) 행렬
    """
    data = np.array(data).astype(np.float64)
    
    if len(data) == feature_num:
        data = data.T
    
    m = np.mean(data, axis=0).astype(np.float32)
    data -= m
    n = len(data) -1 
    
    return data.T @ data / n