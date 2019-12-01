import unittest
from Functions import *
import numpy as np

class OperatorA_tests(unittest.TestCase):
    
    """ Check the action of integral operator A """
        
    def test_empty(self):
        """ Polynomail is empty test """
        print("id : " + self.id())
        poly1 = operatorA_action(np.poly1d([]))
        poly2 = operatorA_action(np.poly1d([0.,0.,0.]))
        
        self.assertEqual(poly1,np.poly1d([0.]))
        self.assertEqual(poly2.coeffs, 0)
        self.assertEqual(poly2.order, 0)
        
    def test_higher_cofficient_zero(self):
        
        """ Does the higher coefficient not equal to zero """
        print("id : " + self.id())
            
        poly0 = np.poly1d([0.,0.,0.,0.,0.,0.,1.,0.])
        poly = operatorA_action(poly0)
        res = np.poly1d([-0.16666667, 0., 0.16666667, 0.])
        
        self.assertEqual(poly.__str__(), res.__str__())
        self.assertEqual(poly.order, 3)
        self.assertEqual(poly0.order, 1)
       
    def test_action_A(self):
        
        """ Test the following polynomials with given V0 """
        print("id : " + self.id())
        
        poly0 = np.poly1d([1.,0.])
        poly1 = operatorA_action(poly0)
        poly2 = operatorA_action(poly1)
        poly3 = operatorA_action(poly2)
        
        res1 = np.poly1d([-0.16666667, 0., 0.16666667, 0.])
        res2 = np.poly1d([0.00833333, 0., -0.0277778, 0., 0.0194444, 0.])
        res3 = np.poly1d([-0.00019841, 0., 0.00138889,  0., -0.00324074, 0., 0.00205026, 0.])
        
        self.assertEqual(poly1.__str__(), res1.__str__())
        self.assertEqual(poly1.order, 3)
        
        self.assertEqual(poly2.__str__(), res2.__str__())
        self.assertEqual(poly2.order, 5)
        
        self.assertEqual(poly3.__str__(), res3.__str__())
        self.assertEqual(poly3.order, 7)

class Sequence_V_tests(unittest.TestCase):
    
    """Sequence_Vector Tests"""
    
    def test_V0_isNull(self):
        print("id : " + self.id())
        a_poly = np.poly1d([0])
        list_of_poly = sequence_V(a_poly,3)
        res = np.poly1d([0.])
        self.assertEqual(list_of_poly[2].__str__(),res.__str__())

    def test_V0_isBig(self):
        print("id : " + self.id())
        a_poly = np.poly1d([1.,1.,1.,1.])
        list_of_poly = sequence_V(a_poly,3)
        res = np.poly1d([ 0.00119048,  0.00277778, 0.00833333, 0.04166667, -0.13333333, 0., 0.07936508, 0. ])
        self.assertEqual(list_of_poly[2].__str__(),res.__str__())

    def test_V0_isConstant(self):
        print("id : " + self.id())
        a_poly = np.poly1d([0,1])
        list_of_poly = sequence_V(a_poly,3)
        res = np.poly1d([ 0.04166667, -0.08333333, 0., 0.04166667, 0.])
        self.assertEqual(list_of_poly[2].__str__(),res.__str__())

                
class ScalarProducts_tests(unittest.TestCase):

    """Scalar Products Tests"""
    
    def test_constants_poly(self):
        print("id : " + self.id())
        a_poly = np.poly1d([0,2])
        b_poly = np.poly1d([0,3])
        self.assertEqual(scalar_product(a_poly,b_poly),6)

    def test_big_poly(self):
        print("id : " + self.id())
        a_poly = np.poly1d([1,2,1,1,2])
        b_poly = np.poly1d([1,2,3,1,3])
        self.assertEqual(scalar_product(a_poly,b_poly),13327/630)

    def test_isNull_poly(self):
        print("id : " + self.id())
        a_poly = np.poly1d([0])
        b_poly = np.poly1d([0])
        self.assertEqual(scalar_product(a_poly,b_poly),0)

    def test_bigDegree_poly(self):
        print("id : " + self.id())
        a_poly = np.poly1d([2,3,0,0,0,0,0,0,0,0,0])
        b_poly = np.poly1d([4,4,0,0,0,0,0,0,0,0,0])
        
        self.assertTrue(scalar_product(a_poly,b_poly)//2==1)
        
class Matrix_tests(unittest.TestCase):

    """Matrix Tests"""
    
    def test_V0_isConstant(self):
        print("id : " + self.id())
        x = get_Matrix(sequence_V(np.poly1d([1]),5),5)
        res = 8.65550344717012e-06
        self.assertTrue(x[0,3] == x[3,0])
        self.assertTrue(x[0,3] == res)

    def test_V0_isNull(self):
        print("id : " + self.id())
        x = get_Matrix(sequence_V(np.poly1d([0]),5),5)
        res = 0
        self.assertTrue(x[0,3] == x[3,0])
        self.assertTrue(x[0,3] == res)

    def test_simple_matrix(self):
        print("id : " + self.id())
        x = get_Matrix(sequence_V(np.poly1d([1,0]),5),5)
        res = 2.1644042808063976e-06
        self.assertTrue(x[0,3] == x[3,0])
        self.assertTrue(x[0,3] == res)
        
    def test_symmetry_matrix(self):
        print("id : " + self.id())
        a_poly = np.poly1d([1,2,1,1,2])
        sec = sequence_V(a_poly, len(a_poly))
        Matrix = get_Matrix(sec, len(sec))
        self.assertTrue(Matrix[0][1]==Matrix[1][0])
        self.assertTrue(Matrix[2][1]+0.000000000000000003 == Matrix[3][0])
        self.assertTrue(Matrix[2][3]==Matrix[3][2])
        self.assertTrue(Matrix[2][1]+0.000000000000000003 ==Matrix[3][0]) #0.000000000000000003 - correction of rounding error

class Get_mu_tests(unittest.TestCase):

    '''Mu Getter Tests'''
    
    def test_quadratic_equation(self):
        print("id : " + self.id())
        x = get_mu(np.array([1,-2,1]))
        res = np.array([1.,1.])
        self.assertEqual(x.__str__(),res.__str__())

    def test_isNull_poly(self):
        print("id : " + self.id())
        x = get_mu(np.array([0.]))
        res = np.array([])
        self.assertEqual(x.__str__(),res.__str__())

    def test_isBig_degree(self):
        print("id : " + self.id())
        x = get_mu(np.array([1,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0]))
        res = np.array([-0.5])
        self.assertEqual(x.__str__(),res.__str__())

class Get_z_tests(unittest.TestCase):

    '''Z Getter Tests'''
    
    def test_V0_isBig(self):
        print("id : " + self.id())
        list_of = sequence_V(np.poly1d([1,1,1]),5)
        x = get_z(np.array([1,2,5,1,2]),list_of,5)
        res = np.poly1d([0.00277778, 0.00833333, -0.125,-0.45833333,-1.,1.57222222, 0.])
        self.assertEqual(x[1].__str__(),res.__str__())

    def test_V0_isNull(self):
        print("id : " + self.id())
        list_of = sequence_V(np.poly1d([0]),5)
        x = get_z(np.array([1,2,5,1,2]),list_of,5)
        res = np.poly1d([0.])
        self.assertEqual(x[4].__str__(),res.__str__())

    def test_Ci_isNull(self):
        print("id : " + self.id())
        list_of = sequence_V(np.poly1d([0]),5)
        x = get_z(np.array([0,0,0,0,0]),list_of,5)
        res = np.poly1d([0.])
        self.assertEqual(x[4].__str__(),res.__str__())
        
        
class Get_u_tests(unittest.TestCase):

    '''U Getter Tests'''
    
    def test_V0_isBig(self):
        print("id : " + self.id())
        x = get_u(np.array([1,1,1,1]),np.array([2,0,3,1]),4)
        self.assertTrue(x[2] == 120)

    def test_z_isNull(self):
        print("id : " + self.id())
        x = get_u(np.array([0,0,0,0]),np.array([8,0,0,1]),4)
        for i in x:
            self.assertTrue(i == 0)

    def test_mu_isNull(self):
        print("id : " + self.id())
        x = get_u(np.array([8,0,0,1]),np.array([0,0,0,0]),4)
        for i in x:
            self.assertTrue(i == 0)

class Vector_res_tests(unittest.TestCase):

    """Vector results Tests"""
    
    def test_not_empty_res_vect(self):
        print("id : " + self.id())
        b = np.poly1d([1,2,1,1,2,6,7,9])
        sec = sequence_V(b, len(b))
        V = len(get_Res_vector_for_Matr(sec,len(sec)))

        self.assertTrue(V != None)
    
    def test_vector_less_then_zero(self):
        print("id : " + self.id())
        b = np.poly1d([1,7,8,9])
        sec = sequence_V(b, len(b))
        V = get_Res_vector_for_Matr(sec,len(sec))

        self.assertTrue(all(V<0))
        
    def test_vector_correctness__of_elements(self):
        print("id : " + self.id())
        b = np.poly1d([1,2,3,4,7])
        sec = sequence_V(b, len(b))
        V = get_Res_vector_for_Matr(sec,len(sec))
        
        self.assertEqual(V[0], -0.009281950772409446)
        self.assertEqual(V[1], -0.09168018527988744)
        self.assertEqual(V[2], -0.9085480722980707)
        self.assertEqual(V[3], -9.213596681096679)

class Vector_Coef_tests(unittest.TestCase):

    """Vector coefs Tests"""
    
    def test_not_empty_coefs(self):
        print("id : " + self.id())
        p = np.poly1d([10,11,12,17,18])
        sec = sequence_V(p, len(p))
        Coefs = get_Vector_of_Coef(sec,len(sec))

        self.assertTrue(len(Coefs) != None)
        
    def test_vector_correctness__of_elements(self):
        print("id : " + self.id())
        p = np.poly1d([1,2,3,4,7])
        sec = sequence_V(p, len(p))
        Coefs = get_Vector_of_Coef(sec,len(sec))
        
        self.assertEqual(Coefs[0], -32.78972631377902)
        self.assertEqual(Coefs[1], -3.330141078744718)
        self.assertEqual(Coefs[2], -0.34217639380311116)
        self.assertEqual(Coefs[3], -0.03209511350429077)
    
    def test_correctness__coefs_func_work(self):
        print("id : " + self.id())
        p = np.poly1d([1,2,3,4,7])
        sec = sequence_V(p, len(p))
        Coefs = get_Vector_of_Coef(sec,len(sec))
        Correct = np.linalg.solve(get_Matrix(sec,len(sec)),get_Res_vector_for_Matr(sec,len(sec)))

        self.assertTrue(all(Coefs == Correct))
