import tensorflow as tf


class BinaryConnectTest(tf.test.TestCase):
    def testBinaryConnect(self):
        bc_module = tf.load_op_library('binary_connect.so')
        with self.test_session():
            result = bc_module.binary_connect([5, -5, 1, 0])
            self.assertAllEqual(result.eval(), [1, -1, 1, 0])
