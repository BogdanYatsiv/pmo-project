import numpy as np

def operatorA_action(p:np.poly1d)->np.poly1d:
    """ Integral operator action A """
    t = np.array([1,0]) #t
    t1 = np.array([-1,1]) #1-t

    first_int = np.polyint(np.polymul(t,p)) # інтегруємо першу частину
    first_addition = np.polymul(t1, first_int) # знаходимо перший доданок

    second_int = np.polyint(np.polymul(t1,p)) # знаходимо первісну
    second_addition = np.polymul(t, np.polysub(np.polyval(second_int,1), second_int)) # підставляємо межі інтегрування та знаходимо другий доданок

    Vm = np.polyadd(first_addition, second_addition) #сумуємо два доданки

    return Vm

def sequence_V(V0:np.poly1d, m:int)->list:
    """ The function generates the sequence Vm """
    sequence = []
    sequence.append(V0) # додаємо початкове наближення
    for i in range (m):
        sequence.append(operatorA_action(sequence[i])) # додаємо поліном в послідовність
    return sequence

def scalar_product(a_poly:np.poly1d, b_poly:np.poly1d)->float:
    """Find scalar product of two polynomials"""
    integral = np.polyint(np.polymul(a_poly, b_poly)) # множимо поліноми і знаходимо первісну
    return integral(1) - integral(0) # від інтегралу в точці 1 - інтеграл в точці 0

def get_Matrix(V_m,num_of_v):
    counter_for_num_of_element = 1
    res = ""
    for counter in range(num_of_v - 1):
        counter_for_el = counter + 1
        if counter == 0:
            matrix_of_SCALAR = np.identity(num_of_v - 1)
        for i in range(counter,num_of_v - 1):
            if counter_for_el  < num_of_v:
                matrix_of_SCALAR[counter,i] = scalar_product(V_m[counter_for_num_of_element],V_m[counter_for_el])
            counter_for_el += 1
        counter_for_num_of_element+=1
        res = matrix_of_SCALAR
    res = symetrize(res)
    res = np.rot90(res,2)
    return res

def symetrize(a):
    return a + a.T - np.diag(a.diagonal())

def get_Res_vector_for_Matr(V_m,num_of_v):
    res = np.array([])
    for counter in range(1,num_of_v):
        res = np.append(res, -(scalar_product(V_m[0],V_m[counter])))
    res = np.flip(res)
    return res

def get_Vector_of_Coef(V_m,num_of_v):
    vector_of_Coef = np.linalg.solve(get_Matrix(V_m,num_of_v),get_Res_vector_for_Matr(V_m,num_of_v))
    return vector_of_Coef

def get_mu(c_arr:np.ndarray)-> np.ndarray:
    """Find mu"""
    mu_poly=c_arr[::-1] # від найбільшого до найменшого
    mu_values=np.roots(mu_poly) # знаходимо корені
    return mu_values

def get_z(c: np.ndarray, v: list, amount: int) -> np.ndarray:
    """Find z"""
    result = np.arange(amount, dtype=np.poly1d)  # створюємо масив поліномів
    for j in range(1, amount + 1):  # знаходимо усі z згідно формули
        val = 0
        for i in range(j):
            val += np.poly1d(c[i] * v[j - i])
        result[j - 1] = val
    return result

def get_u(z: np.ndarray, mu: np.ndarray, amount: int)->np.ndarray:
    """Find u """
    n_max = len(mu)
    result = np.arange(n_max, dtype=np.poly1d)
    for n in range(n_max):
        val = 0
        for j in range(amount):
            val += z[j] * pow(mu[n], j + 1)
        result[n] = val
    return result
