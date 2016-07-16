from __future__ import print_function

from pprint import pprint

import numpy as np

from interpreter import Interpreter

s_expr = ('bundle',
          ('constants',
           ('$array_of_int32_type',
            ('python-type', '&array_of_int32TypeObject'))),
          ('func', '$foo',   # added name here
              ('signature', ('param', '$array_of_int32_type', '$a'), ('result', 'int32')),
              ('local', '$total', 'i32'),
              # question: are these zero-initialized or undefined?
              ('local', '$i', 'i64'),
              ('set_local', '$total', ('i32.const', 0)),
              ('set_local', '$i', ('i64.const', 0)),
              ('loop', '$done', '$loop',
               ('if',
                ('i64.ge', ('get_local', '$i'), ('i64.load', '$a', 24)),
                   ('br', '$done'),
                   ('block',
                    ('set_local',
                     '$total',
                     ('i32.add',
                      ('get_local', '$total'),
                         ('i32.load',
                          ('i64.add',
                           ('i64.load', '$a', 16),
                              ('i64.mul', ('i64.const', 4), (('get_local', '$i')))),
                          0))),
                    ('set_local', '$i', ('i64.add', ('get_local', '$i'), ('i64.const', 1))))),
                  ('br', '$loop')),
              ('get_local', '$total')),
          )

#
#
#

# pprint(s_expr)

interp = Interpreter(s_expr)

mem = interp.memory

# allocate ndarray structure
#  offset 0: dummy
#  offset 16: data pointer
#  offset 24: shape pointer
#  size 32 bytes
ary_struct_ptr = mem.allocate(32)
print(hex(ary_struct_ptr))
ary_struct_i64 = mem.get_buffer(ary_struct_ptr, astype=np.uint64)
arysize = 10
ary_struct_i64[2] = dataptr = mem.allocate(
    arysize * np.dtype('int32').itemsize)
ary_struct_i64[3] = arysize


data = mem.get_buffer(dataptr, astype=np.uint32)
data[:] = np.arange(arysize) + 1

frozen_data = data.copy()
print('data', data)

# run foo
result = interp.run("$foo", args=[np.int64(ary_struct_ptr)])

expect = frozen_data.sum()
print('result', result)

assert expect == result
assert np.all(frozen_data == data)
