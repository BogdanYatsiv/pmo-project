import numpy as np
from Functions import *
#TODO:task1, запрограмувати дію оператора А(Перевірити чи старший коефіцієнт != 0)
start = np.poly1d([1.,0.])
first_step = operatorA_action(start)
second_step = sequence_V(first_step,3)
#TODO:task2, обчислити скалярний добуток (Vi,Vj)
#TODO:task3, розв'язати систему лінійних алгебраїчних рівнянь(використати np.linalg?)
third_step = get_Vector_of_Coef(second_step,3)
#TODO:task4, побудова власних функцій U задопомогою numpy
fourth_step = get_mu(third_step)
fifth_step = get_z(third_step,second_step,2)
last_step = get_u(fifth_step,fourth_step,2)
print(last_step)