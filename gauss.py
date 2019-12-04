import numpy as np
def gauss(M,V):
    first_step = gauss_Elimination(M,V)
    second_step = gauss_Back_Substitution(first_step[0],first_step[1])
    return second_step

def gauss_Elimination(M,V):
    for k in range(len(V)-1):
        for i in range(k + 1,len(V)):
            if M[i][k] == 0:
                continue
            mult_var = M[k][k]/M[i][k]
            for j in range(k,len(V)):
                M[i][j] = M[k][j] - M[i][j]*mult_var     
            M[i][i] /= M[i][i] 
            V[i]= V[k] - V[i]*mult_var
    print(M)
    print(V)
    return M,V

def gauss_Back_Substitution(M,V):    
    size = len(V)
    X = np.zeros(size)
    X[size-1]=V[size-1]/M[size-1][size-1]
    for i in range(size - 2, -1, -1):
        sum_of_matrix_row = 0
        for j in range(i+1,size):
            sum_of_matrix_row += M[i][j]*X[j] 
        X[i]=(V[i] - sum_of_matrix_row)*M[i][i]
    return X