import numpy as np

def operatorA_action(V0:np.poly1d, number:int)->list:
    """ Obtaining successive iterations Vm using the integral operator kernel """
    p = V0 # задаємо початкове наближення
    t = np.array([1,0]) #t
    t1 = np.array([-1,1]) #1-t
    m = 1
    list_V = [] # список поліномів Vm
    list_V.append(V0)

    while m < number:
        first_int = np.polyint(np.polymul(t,p)) # інтегруємо першу частину
        first_addition = np.polymul(t1, first_int) # знаходимо перший доданок

        second_int = np.polyint(np.polymul(t1,p)) # знаходимо первісну
        second_addition = np.polymul(t, np.polysub(np.polyval(second_int,1), second_int)) # підставляємо межі інтегрування та знаходимо другий доданок

        Vm = np.polyadd(first_addition, second_addition) #сумуємо два доданки
        p = Vm # попередній поліном

        list_V.append(Vm)
        m += 1

    return list_V

def scalar_product(a_poly:np.poly1d, b_poly:np.poly1d)->float:
    """Find scalar product of two polynomials"""
    integral = np.polyint(np.polymul(a_poly, b_poly)) # множимо поліноми і знаходимо первісну
    return integral(1) - integral(0) # від інтегралу в точці 1 - інтеграл в точці 0

def get_mu(c_arr:np.ndarray)-> np.ndarray:
    """Find mu"""
    mu_poly=c_arr[::-1] # від найбільшого до найменшого
    mu_values=np.roots(mu_poly) # знаходимо корені
    return mu_values

def get_z(c:np.ndarray, v:list, amount:int)->np.ndarray:
    """Find z"""
    result = np.arange(amount, dtype=np.poly1d) #створюємо масив поліномів
    for j in range(1, amount + 1): # знаходимо усі z згідно формули
        val = 0
        for i in range(j):
            val += c[i] * v[j - i]
        result[j-1] = val
    return result
