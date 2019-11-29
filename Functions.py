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

def scalar_product(a_poly:np.polyld, b_poly:np.polyld)->float:
    """Find scalar product of two polynomials"""
    integral = np.polyint(np.polymul(a_poly, b_poly)) # множимо поліноми і знаходимо первісну
    return integral(1) - integral(0) # від інтегралу в точці 1 - інтеграл в точці 0
