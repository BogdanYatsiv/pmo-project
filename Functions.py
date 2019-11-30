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
    matrix_of_SCALAR = np.array([[]])
    counter_for_num_of_element = 1
    for counter in range(num_of_v - 1):
        if counter == 0:
            matrix_of_SCALAR = np.zeros((1,1))
            matrix_of_SCALAR[counter,counter] = scalar_product(V_m[counter_for_num_of_element],V_m[counter_for_num_of_element])
        elif counter > 0:
            col = np.zeros((counter,1))
            row = np.zeros((counter + 1,1))
            matrix_of_SCALAR = np.concatenate((matrix_of_SCALAR,col), axis = 1)
            matrix_of_SCALAR = np.concatenate((matrix_of_SCALAR,row.T), axis = 0)
            local_count = 0
            matrix_of_SCALAR[counter,counter] = scalar_product(V_m[counter_for_num_of_element],V_m[counter_for_num_of_element])
            while local_count < counter:
                matrix_of_SCALAR[counter,local_count] = scalar_product(V_m[counter],V_m[local_count])
                matrix_of_SCALAR[local_count,counter] = scalar_product(V_m[local_count],V_m[counter])
                local_count+=1
        counter_for_num_of_element+=1
    matrix_of_SCALAR = np.rot90(matrix_of_SCALAR,2)
    return matrix_of_SCALAR

def get_Res_vector_for_Matr(V_m,num_of_v):
    res = np.array([])
    for counter in range(1,num_of_v):
        res = np.append(res, -(scalar_product(V_m[0],V_m[counter])))
    res = np.flip(res)
    return res

def get_Vector_of_Coef(V_m,num_of_v):
    vector_of_Coef = np.linalg.solve(get_Matrix(V_m,num_of_v),get_Res_vector_for_Matr(V_m,num_of_v))
    return vector_of_Coef