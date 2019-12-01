import unittest
import UnitTests

calcTestSuite = unittest.TestSuite()
calcTestSuite.addTest(unittest.makeSuite(UnitTests.ScalarProducts_tests))
calcTestSuite.addTest(unittest.makeSuite(UnitTests.Sequence_V_tests))
calcTestSuite.addTest(unittest.makeSuite(UnitTests.OperatorA_tests))
calcTestSuite.addTest(unittest.makeSuite(UnitTests.Matrix_tests))
calcTestSuite.addTest(unittest.makeSuite(UnitTests.Vector_res_tests))
calcTestSuite.addTest(unittest.makeSuite(UnitTests.Vector_Coef_tests))

runner = unittest.TextTestRunner(verbosity=2)
runner.run(calcTestSuite)
