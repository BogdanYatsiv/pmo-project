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
