from __future__ import print_function

from pprint import pprint

import numpy as np


class Memory(object):
    """
    Memory system for the virtual machine.

    Assume:
     - 64-bit machine (pointer size)
     - 32-bit maximum allocated size (lower bits for pointer)
     - 32-bit maximum number of allocation (upper bits for pointer)
    """
    PTRBITS = 64
    PAGEBITS = 32
    OFFSETBITS = 32
    OFFSETMASK = (2 ** OFFSETBITS) - 1

    def __init__(self):
        self.vpages = {}
        self.numalloc = 0

    def allocate(self, size):
        """
        allocate ``size`` bytes of memory.
        """
        assert not (size >> self.OFFSETBITS), 'allocation too large'
        mem = np.zeros(size, dtype=np.uint8)
        self.numalloc += 1
        pageaddr = self.numalloc
        self.vpages[pageaddr] = mem

        return self.make_pointer(pageaddr, 0)

    def get_buffer(self, ptr, astype=np.uint8):
        """
        Get the underlying numpy array for the pointer
        """
        page, offset = self.parse_pointer(ptr)
        raw = self.vpages[page][offset:]
        return raw.view(dtype=astype)

    def make_pointer(self, pageaddr, offset):
        """
        Return pointer from page-address and byte offset
        """
        return pageaddr << self.OFFSETBITS + offset

    def parse_pointer(self, ptr):
        """
        Convert pointer to page-address and byte offset
        """
        offset = ptr & self.OFFSETMASK
        page = ptr >> self.OFFSETBITS
        return page, offset


def parse_bundle(sexpr):
    # parse bundle for constants and functions
    bp = BundleParser()
    bp.run(sexpr)
    return bp


class BundleParser(object):
    """
    Parse bundle for functions and constants
    """

    def __init__(self):
        self.constants = {}
        self.functions = {}

    def run(self, sexpr):
        opc = sexpr[0]
        fname = 'op_%s' % opc
        fn = getattr(self, fname)
        return fn(sexpr)

    def op_bundle(self, sexpr):
        body = sexpr[1:]
        for elem in body:
            self.run(elem)

    def op_constants(self, sexpr):
        elemlist = sexpr[1:]
        for elem in elemlist:
            self.constants[elem[0]] = elem[1:]

    def op_func(self, sexpr):
        name = sexpr[1]
        self.functions[name] = sexpr


class Frame(object):

    def __init__(self, body):
        self.locals = {}
        self.stack = []
        self.body = body
        self.pc = 0  # programcounter
        self.prev = None
        self.goto = None

    def get_expr_at_pc(self):
        return self.body[self.pc]

    @property
    def is_ended(self):
        return self.pc >= len(self.body) or self.goto is not None

    def incr_pc(self):
        self.pc += 1

    def handle_label(self, interp, label):
        interp.pop_frame()
        return False

    def declare_local(self, name, typ):
        dtype = typestr_to_dtype(typ)
        self.locals[name] = make_scalar(dtype)

    def set_local(self, name, val):
        try:
            var = self.locals[name]
        except KeyError:
            self.prev.set_local(name, val)
        else:
            if var.dtype is not val.dtype:
                fmt = 'var {name} expect {expect} but got {got}'
                raise TypeError(fmt.format(name=name, expect=var.dtype,
                                           got=val.dtype))
            self.locals[name] = val

    def get_local(self, name):
        try:
            return self.locals[name]
        except KeyError:
            if self.prev is not None:
                return self.prev.get_local(name)
            else:
                raise


class BlockFrame(Frame):

    def __init__(self, body):
        super(BlockFrame, self).__init__(body)


class LoopFrame(Frame):

    def __init__(self, label_end, label_body, body):
        super(LoopFrame, self).__init__(body)
        self.label_end = label_end
        self.label_body = label_body

    def handle_label(self, interp, label):
        if label == self.label_end:
            interp.pop_frame()
            return True
        elif label == self.label_body:
            self.pc = 0
            return True
        else:
            return False


_typestr_to_dtype = {
    'i32': np.int32,
    'i64': np.int64,
}


def typestr_to_dtype(typestr):
    if typestr.startswith('$'):
        # XXX assume pointer
        return np.int64
    return _typestr_to_dtype[typestr]


def make_scalar(dtype, init=None):
    dtype = np.dtype(dtype)
    if init is None:
        return dtype.type()
    else:
        return dtype.type(init)


def signature_check(sig, args):
    assert sig[0] == 'signature'
    argtys = sig[1:-1]
    resty = sig[-1]
    argnametys = []
    for at in argtys:
        assert at[0] == 'param'
        argnametys.append((at[2], at[1]))
    assert resty[0] == 'result'
    assert len(args) == len(argnametys)
    return argnametys


class Interpreter(object):
    """
    Interprets a bundle.
    XXX: starts with one function call first
    """

    def __init__(self, sexpr):
        self.memory = Memory()
        pb = parse_bundle(sexpr)
        self.constants = pb.constants
        self.functions = pb.functions
        self.frame = None

    def run(self, fname, args):
        func = self.functions[fname]
        assert fname == func[1]
        sig = func[2]
        body = func[3:]
        frame = Frame(body)
        # quick signature check
        for (aname, atype), aval in zip(signature_check(sig, args), args):
            frame.declare_local(aname, atype)
            frame.set_local(aname, aval)
        # set frame
        self.push_frame(frame)
        return self.eval_loop()

    def push_frame(self, frame):
        assert frame.prev is None
        frame.prev = self.frame
        self.frame = frame

    def pop_frame(self):
        self.frame = self.frame.prev

    def eval_loop(self):
        tos = None
        while self.frame:
            sexpr = self.frame.get_expr_at_pc()
            self.frame.incr_pc()
            tos = self.eval_expr(sexpr)
            if self.frame.is_ended:
                target = self.frame.goto
                # handle branching by label
                if target is None:
                    self.pop_frame()

                else:
                    while not self.frame.handle_label(self, target):
                        pass
        return tos

    def eval_expr(self, sexpr):
        print('EVAL', sexpr)
        opc = sexpr[0].replace('.', '_')
        fname = 'op_%s' % opc
        fn = getattr(self, fname)
        return fn(sexpr)

    def op_local(self, sexpr):
        _, name, typ = sexpr
        self.frame.declare_local(name, typ)

    def op_set_local(self, sexpr):
        _, name, rhs = sexpr
        val = self.eval_expr(rhs)
        self.frame.set_local(name, val)

    def op_get_local(self, sexpr):
        _, name = sexpr
        return self.frame.get_local(name)

    def op_loop(self, sexpr):
        label_end, label_body = sexpr[1:3]
        body = sexpr[3:]
        frame = LoopFrame(label_end, label_body, body)
        self.push_frame(frame)

    def op_block(self, sexpr):
        body = sexpr[1:]
        frame = BlockFrame(body)
        self.push_frame(frame)

    def op_if(self, sexpr):
        _, cond, then, otherwise = sexpr
        condval = self.eval_expr(cond)
        return self.eval_expr(then if condval else otherwise)

    def op_br(self, sexpr):
        _, target = sexpr
        self.frame.goto = target

    #
    # expressions
    #

    def op_i32_const(self, sexpr):
        _, const = sexpr
        return make_scalar(typestr_to_dtype('i32'), init=const)

    def op_i64_const(self, sexpr):
        _, const = sexpr
        return make_scalar(typestr_to_dtype('i64'), init=const)

    def op_i64_ge(self, sexpr):
        _, lhs, rhs = sexpr
        lval = self.eval_expr(lhs)
        rval = self.eval_expr(rhs)
        return np.uint8(lval >= rval)

    def _op_load(self, sexpr, dtype):
        _, ptr, offset = sexpr
        # XXX: bad IR semantic
        if isinstance(ptr, str):
            ptrval = self.frame.get_local(ptr)
        else:
            ptrval = self.eval_expr(ptr)
        ptrval += + int(offset)
        return self.memory.get_buffer(ptrval, astype=dtype)[0]

    def op_i64_load(self, sexpr):
        return self._op_load(sexpr, np.int64)

    def op_i32_load(self, sexpr):
        return self._op_load(sexpr, np.int32)

    def op_i32_add(self, sexpr):
        _, lhs, rhs = sexpr
        lval = self.eval_expr(lhs)
        rval = self.eval_expr(rhs)
        return lval + rval

    def op_i64_add(self, sexpr):
        _, lhs, rhs = sexpr
        lval = self.eval_expr(lhs)
        rval = self.eval_expr(rhs)
        return lval + rval

    def op_i64_mul(self, sexpr):
        _, lhs, rhs = sexpr
        lval = self.eval_expr(lhs)
        rval = self.eval_expr(rhs)
        return lval * rval
