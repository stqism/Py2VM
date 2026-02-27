from ctypes import *
import dis
import binascii
import sys
import StringIO
import types


def hasarg(opcode):
    if opcode.startswith('STOP'):
        return 0
    elif opcode.startswith('NOP'):
        return 0
    elif opcode == 'POP_TOP':
        return 0
    elif opcode == 'POP_BLOCK':
        return 0
    elif opcode.startswith('ROT'):
        return 0
    elif opcode.startswith('DUP'):
        return 0
    elif opcode.startswith('UNARY'):
        return 0
    elif opcode.startswith('GET'):
        return 0
    elif opcode.startswith('BINARY'):
        return 0
    elif opcode.startswith('INPLACE'):
        return 0
    elif opcode.startswith('PRINT'):
        return 0
    elif opcode.startswith('BREAK'):
        return 0
    elif opcode == 'LOAD_LOCALS':
        return 0
    elif opcode == 'RETURN_VALUE':
        return 0
    elif opcode == 'YIELD_VALUE':
        return 0
    elif opcode == 'IMPORT_STAR':
        return 0
    elif opcode == 'END_FINALLY':
        return 0
    elif opcode == 'BUILD_CLASS':
        return 0
    elif opcode == 'WITH_CLEANUP':
        return 0
    elif opcode == 'EXEC_STMT':
        return 0
    else:
        return 1


def bytecode_optimize(bytecode):
    bytecode_list = []
    offset_to_index = {}
    code = bytecode.co_code
    i = 0
    while i < len(code):
        offset = i
        opcode_byte = ord(code[i])
        i += 1
        if opcode_byte == 0:
            continue
        opcode_value = dis.opname[opcode_byte]
        if hasarg(opcode_value):
            arg = ord(code[i]) | (ord(code[i + 1]) << 8)
            i += 2
            # Convert relative jump offsets to absolute byte offsets
            if opcode_value in ('JUMP_FORWARD', 'FOR_ITER',
                                'SETUP_LOOP', 'SETUP_EXCEPT', 'SETUP_FINALLY'):
                arg = i + arg
        else:
            arg = 0
        offset_to_index[offset] = len(bytecode_list)
        bytecode_list.append([opcode_value, arg])
    return bytecode_list, offset_to_index


def buildcode(code):
    return py2vm(compile(code, '<none>', 'exec'))


def py2vm(bytecode, stack=False, rec_log=False, fast_locals=None):

    if rec_log != False:
        log = rec_log
    else:
        log = StringIO.StringIO()
        log.write('py2vm output:\n')

    if stack != False:
        const_stack = stack
    else:
        const_stack = []

    fast_dict = fast_locals if fast_locals is not None else {}

    jump = 0

    block_stack = []

    __INTERNAL__DEBUG_LOG = 1
    __INTERNAL__DEBUG_LOG_CONST = 0
    __INTERNAL__DEBUG_LOG_VAR = 0
    __INTERNAL__UNSAFE_FUNCTION = 0
    name_dict = {}

    # I was inspired by str8C
    i = -1
    opcode_array, offset_to_index = bytecode_optimize(bytecode)
    while i < len(opcode_array):
        i += 1

        opcode_pack = opcode_array[i]
        opcode_value = opcode_pack[0]
        arg = opcode_pack[1]

        # log.write(const_stack)
        # log.write(opcode_value)

        if __INTERNAL__DEBUG_LOG_CONST:
            log.write(str(const_stack) + '\n')

        if __INTERNAL__DEBUG_LOG_VAR:
            log.write(str(name_dict) + '\n')

        if opcode_value == 'NOP':
            if __INTERNAL__DEBUG_LOG:
                log.write('DEBUG => internal placeholder\n')

        elif opcode_value == 'POP_TOP':
            if const_stack:
                del const_stack[0]

            if __INTERNAL__DEBUG_LOG:
                log.write('DEBUG => removed top of stack\n')

        elif opcode_value == 'ROT_TWO':
            stk2 = const_stack.pop(0)
            const_stack.insert(1, stk2)

            if __INTERNAL__DEBUG_LOG:
                log.write('DEBUG => rotated top two stack items\n')

        elif opcode_value == 'ROT_THREE':
            stk2 = const_stack.pop(0)
            const_stack.insert(2, stk2)

            if __INTERNAL__DEBUG_LOG:
                log.write('DEBUG => moved top of stack down 2\n')

        elif opcode_value == 'ROT_FOUR':
            stk2 = const_stack.pop(0)
            const_stack.insert(3, stk2)

            if __INTERNAL__DEBUG_LOG:
                log.write('DEBUG => moved top of stack down 3\n')

        elif opcode_value == 'DUP_TOP':
            const_stack.insert(0, const_stack[0])

            if __INTERNAL__DEBUG_LOG:
                log.write('DEBUG => duplicated top of stack\n')

        elif opcode_value == 'LOAD_CONST':
            const_stack.insert(0, bytecode.co_consts[arg])

            if __INTERNAL__DEBUG_LOG:
                log.write(
                    'DEBUG => loaded %s on to stack\n' % (bytecode.co_consts[arg]))

        elif opcode_value == 'LOAD_FAST':
            var_name = bytecode.co_varnames[arg]
            const_stack.insert(0, fast_dict.get(var_name))

            if __INTERNAL__DEBUG_LOG:
                log.write(
                    'DEBUG => loaded %s (%s) on to stack\n' % (var_name, fast_dict.get(var_name)))

        elif opcode_value == 'LOAD_NAME':
            const_stack.insert(0, name_dict.get(arg))

            if __INTERNAL__DEBUG_LOG:
                log.write('DEBUG => loaded %s on to stack\n' %
                          (name_dict.get(arg)))

        elif opcode_value == 'STORE_FAST':
            fast_dict[bytecode.co_varnames[arg]] = const_stack.pop(0)

            if __INTERNAL__DEBUG_LOG:
                log.write("DEBUG => stored %s into %s\n" % (
                    fast_dict[bytecode.co_varnames[arg]], bytecode.co_varnames[arg]))

        elif opcode_value == 'STORE_NAME':
            name_dict[arg] = const_stack.pop(0)
            if bytecode.co_names[arg] == '__INTERNAL__DEBUG_LOG':
                __INTERNAL__DEBUG_LOG = name_dict[arg]

                if __INTERNAL__DEBUG_LOG:
                    log.write(
                        "DEBUG => tripped internal debugger: verbose output\n")

            elif bytecode.co_names[arg] == '__INTERNAL__DEBUG_LOG_CONST':
                __INTERNAL__DEBUG_LOG_CONST = name_dict[arg]

                if __INTERNAL__DEBUG_LOG:
                    log.write(
                        "DEBUG => tripped internal debugger: const print\n")

            elif bytecode.co_names[arg] == '__INTERNAL__DEBUG_LOG_VAR':
                __INTERNAL__DEBUG_LOG_VAR = name_dict[arg]

                if __INTERNAL__DEBUG_LOG:
                    log.write(
                        "DEBUG => tripped internal debugger: name dict print\n")

            elif bytecode.co_names[arg] == '__INTERNAL__UNSAFE_FUNCTION':
                __INTERNAL__UNSAFE_FUNCTION = name_dict[arg]

                if __INTERNAL__DEBUG_LOG:
                    log.write(
                        "DEBUG => tripped unsafe function support\n")

            if __INTERNAL__DEBUG_LOG:
                log.write("DEBUG => set value of var %s to %s\n" %
                          (bytecode.co_names[arg], name_dict[arg]))

        elif opcode_value == 'DELETE_NAME':
            del name_dict[arg]

            if __INTERNAL__DEBUG_LOG:
                log.write("DEBUG => deleted name dict entry %s\n" % (arg))

        elif opcode_value == 'UNARY_POSITIVE':
            const_stack.insert(0, +(const_stack[0]))

            if __INTERNAL__DEBUG_LOG:
                log.write("DEBUG => set top of stack to +(tos)\n")

        elif opcode_value == 'UNARY_NEGATIVE':
            const_stack.insert(0, -(const_stack[0]))

            if __INTERNAL__DEBUG_LOG:
                log.write("DEBUG => set top of stack to -(tos)\n")

        elif opcode_value == 'UNARY_NOT':
            const_stack.insert(0, not const_stack[0])

            if __INTERNAL__DEBUG_LOG:
                log.write("DEBUG => set top of stack to not tos\n")

        elif opcode_value == 'UNARY_CONVERT':
            const_stack.insert(0, repr(const_stack[0]))

            if __INTERNAL__DEBUG_LOG:
                log.write("DEBUG => added repr(tos) to top of stack\n")

        elif opcode_value == 'UNARY_INVERT':
            const_stack.insert(0, ~const_stack[0])

            if __INTERNAL__DEBUG_LOG:
                log.write("DEBUG => added ~tos to top of stack\n")

        elif opcode_value == 'GET_ITER':
            const_stack[0] = iter(const_stack[0])

            if __INTERNAL__DEBUG_LOG:
                log.write("DEBUG => replaced tos with iter(tos)\n")

        elif opcode_value == 'BINARY_POWER':
            # We're supposed to remove tos/tos1 on binary_calls
            math0 = const_stack.pop(0)
            math1 = const_stack.pop(0)
            const_stack.insert(0, math1 ** math0)

            if __INTERNAL__DEBUG_LOG:
                log.write("DEBUG => got power of %s and %s\n" % (math1, math0))

        elif opcode_value == 'BINARY_MULTIPLY':
            math0 = const_stack.pop(0)
            math1 = const_stack.pop(0)
            const_stack.insert(0, math1 * math0)

            if __INTERNAL__DEBUG_LOG:
                log.write("DEBUG => multiplied %s with %s\n" % (math0, math1))

        elif opcode_value == 'BINARY_DIVIDE':
            math0 = const_stack.pop(0)
            math1 = const_stack.pop(0)
            const_stack.insert(0, math1 / math0)

            if __INTERNAL__DEBUG_LOG:
                log.write("DEBUG => divided %s with %s\n" % (math0, math1))

        elif opcode_value == 'BINARY_FLOOR_DIVIDE':
            math0 = const_stack.pop(0)
            math1 = const_stack.pop(0)
            const_stack.insert(0, math1 // math0)

            if __INTERNAL__DEBUG_LOG:
                log.write("DEBUG => divided %s with %s\n" % (math0, math1))

        elif opcode_value == 'BINARY_MODULO':
            math0 = const_stack.pop(0)
            math1 = const_stack.pop(0)
            const_stack.insert(0, math1 % math0)

            if __INTERNAL__DEBUG_LOG:
                log.write("DEBUG => got mod of %s and %s\n" % (math0, math1))

        elif opcode_value == 'BINARY_TRUE_DIVIDE':
            math0 = const_stack.pop(0)
            math1 = const_stack.pop(0)
            const_stack.insert(0, math1 / math0)

            if __INTERNAL__DEBUG_LOG:
                log.write("DEBUG => divided %s with %s\n" % (math0, math1))

        elif opcode_value == 'BINARY_ADD':
            math0 = const_stack.pop(0)
            math1 = const_stack.pop(0)
            const_stack.insert(0, math1 + math0)

            if __INTERNAL__DEBUG_LOG:
                log.write("DEBUG => added %s with %s\n" % (math0, math1))

        elif opcode_value == 'BINARY_SUBTRACT':
            math0 = const_stack.pop(0)
            math1 = const_stack.pop(0)
            const_stack.insert(0, math1 - math0)

            if __INTERNAL__DEBUG_LOG:
                log.write("DEBUG => subtracted %s from %s\n" % (math1, math0))

        elif opcode_value == 'BINARY_SUBSCR':
            math0 = const_stack.pop(0)
            math1 = const_stack.pop(0)
            const_stack.insert(0, math1[math0])

            if __INTERNAL__DEBUG_LOG:
                log.write("DEBUG => set tos to %s[%s]\n" % (math1, math0))

        elif opcode_value == 'BINARY_LSHIFT':
            math0 = int(const_stack.pop(0))
            math1 = int(const_stack.pop(0))
            const_stack.insert(0, math1 << math0)

            if __INTERNAL__DEBUG_LOG:
                log.write("DEBUG => shifted %s left %s\n" % (math1, math0))

        elif opcode_value == 'BINARY_RSHIFT':
            math0 = int(const_stack.pop(0))
            math1 = int(const_stack.pop(0))
            const_stack.insert(0, math1 >> math0)

            if __INTERNAL__DEBUG_LOG:
                log.write("DEBUG => shifted %s right %s\n" % (math1, math0))

        elif opcode_value == 'BINARY_AND':
            math0 = int(const_stack.pop(0))
            math1 = int(const_stack.pop(0))
            const_stack.insert(0, math1 & math0)

            if __INTERNAL__DEBUG_LOG:
                log.write("DEBUG => %s AND %s\n" % (math1, math0))

        elif opcode_value == 'BINARY_XOR':
            math0 = int(const_stack.pop(0))
            math1 = int(const_stack.pop(0))
            const_stack.insert(0, math1 ^ math0)

            if __INTERNAL__DEBUG_LOG:
                log.write("DEBUG => %s XOR %s\n" % (math1, math0))

        elif opcode_value == 'BINARY_OR':
            math0 = int(const_stack.pop(0))
            math1 = int(const_stack.pop(0))
            const_stack.insert(0, math1 | math0)

            if __INTERNAL__DEBUG_LOG:
                log.write("DEBUG => %s OR %s\n" % (math1, math0))

        elif opcode_value == 'COMPARE_OP':
            math0 = const_stack.pop(0)
            math1 = const_stack.pop(0)
            op_name = dis.cmp_op[arg]
            if op_name == '<':
                result = math1 < math0
            elif op_name == '<=':
                result = math1 <= math0
            elif op_name == '==':
                result = math1 == math0
            elif op_name == '!=':
                result = math1 != math0
            elif op_name == '>':
                result = math1 > math0
            elif op_name == '>=':
                result = math1 >= math0
            elif op_name == 'in':
                result = math1 in math0
            elif op_name == 'not in':
                result = math1 not in math0
            elif op_name == 'is':
                result = math1 is math0
            elif op_name == 'is not':
                result = math1 is not math0
            else:
                result = False
            const_stack.insert(0, result)

            if __INTERNAL__DEBUG_LOG:
                log.write("DEBUG => compared %s %s %s => %s\n" % (math1, op_name, math0, result))

        elif opcode_value == 'POP_JUMP_IF_FALSE':
            tos = const_stack.pop(0)
            if not tos:
                i = offset_to_index[arg] - 1

            if __INTERNAL__DEBUG_LOG:
                log.write("DEBUG => POP_JUMP_IF_FALSE: tos=%s target=%s\n" % (tos, arg))

        elif opcode_value == 'POP_JUMP_IF_TRUE':
            tos = const_stack.pop(0)
            if tos:
                i = offset_to_index[arg] - 1

            if __INTERNAL__DEBUG_LOG:
                log.write("DEBUG => POP_JUMP_IF_TRUE: tos=%s target=%s\n" % (tos, arg))

        elif opcode_value == 'JUMP_IF_TRUE_OR_POP':
            if const_stack[0]:
                i = offset_to_index[arg] - 1
            else:
                del const_stack[0]

            if __INTERNAL__DEBUG_LOG:
                log.write("DEBUG => JUMP_IF_TRUE_OR_POP: target=%s\n" % arg)

        elif opcode_value == 'JUMP_IF_FALSE_OR_POP':
            if not const_stack[0]:
                i = offset_to_index[arg] - 1
            else:
                del const_stack[0]

            if __INTERNAL__DEBUG_LOG:
                log.write("DEBUG => JUMP_IF_FALSE_OR_POP: target=%s\n" % arg)

        elif opcode_value == 'JUMP_FORWARD':
            i = offset_to_index[arg] - 1

            if __INTERNAL__DEBUG_LOG:
                log.write("DEBUG => jumped forward to %s\n" % (arg))

        elif opcode_value == 'JUMP_ABSOLUTE':
            i = offset_to_index[arg] - 1

            if __INTERNAL__DEBUG_LOG:
                log.write("DEBUG => jumped to %s\n" % (arg))

        elif opcode_value == 'SETUP_LOOP':
            block_stack.append(('loop', arg))

            if __INTERNAL__DEBUG_LOG:
                log.write("DEBUG => setup loop, exit at byte %s\n" % arg)

        elif opcode_value == 'POP_BLOCK':
            if block_stack:
                block_stack.pop()

            if __INTERNAL__DEBUG_LOG:
                log.write("DEBUG => popped block\n")

        elif opcode_value == 'BREAK_LOOP':
            if block_stack:
                _, exit_offset = block_stack.pop()
                i = offset_to_index[exit_offset] - 1

            if __INTERNAL__DEBUG_LOG:
                log.write("DEBUG => break loop\n")

        elif opcode_value == 'FOR_ITER':
            try:
                val = next(const_stack[0])
                const_stack.insert(0, val)
            except StopIteration:
                const_stack.pop(0)
                i = offset_to_index[arg] - 1

            if __INTERNAL__DEBUG_LOG:
                log.write("DEBUG => FOR_ITER\n")

        elif opcode_value == 'PRINT_ITEM':
            log.write(str(const_stack.pop(0)))

        elif opcode_value == 'PRINT_NEWLINE':
            log.write('\n')

        elif opcode_value == 'LOAD_ATTR':
            math0 = const_stack.pop(0)
            const_stack.insert(0, getattr(math0, bytecode.co_names[arg]))

        elif opcode_value == 'MAKE_FUNCTION':
            pass  # code object stays on TOS; STORE_NAME will consume it

        elif opcode_value == 'CALL_FUNCTION':
            argc = arg & 0xff
            kwargc = (arg >> 8) & 0xff

            kwargs = {}
            for _ in xrange(kwargc):
                val = const_stack.pop(0)
                key = const_stack.pop(0)
                kwargs[key] = val

            args_list = []
            for _ in xrange(argc):
                args_list.insert(0, const_stack.pop(0))

            func = const_stack.pop(0)

            if __INTERNAL__UNSAFE_FUNCTION:
                func_name = func if isinstance(func, str) else getattr(func, '__name__', repr(func))
                obj = compile("%s(%s)" % (func_name, str(args_list).strip('[]')), '<string>', 'eval')
                result = eval(obj)
                const_stack.insert(0, result)

                if __INTERNAL__DEBUG_LOG:
                    log.write("DEBUG => UNSAFE FUNCTION CALL: %s(%s) => %s\n" % (func_name, args_list, result))

            elif isinstance(func, types.CodeType):
                fl = {}
                for idx, val in enumerate(args_list):
                    if idx < len(func.co_varnames):
                        fl[func.co_varnames[idx]] = val
                result_stack, log = py2vm(func, [], log, fast_locals=fl)
                const_stack.insert(0, result_stack[0] if result_stack else None)

                if __INTERNAL__DEBUG_LOG:
                    log.write("DEBUG => called function %s(%s)\n" % (func.co_name, args_list))

            else:
                try:
                    result = func(*args_list, **kwargs)
                    const_stack.insert(0, result)
                except Exception as e:
                    log.write("ERROR => could not call %s: %s\n" % (func, e))
                    const_stack.insert(0, None)

        elif opcode_value == 'RETURN_VALUE':

            if __INTERNAL__DEBUG_LOG:
                log.write("DEBUG => returned execution back to parent\n")
            break

        else:
            log.write("unknown opcode: %s\n" % (opcode_value))

    i += 1

    if stack != False:
        return stack, log
    else:
        return log.getvalue()


code = """
__INTERNAL__DEBUG_LOG=1
__INTERNAL__UNSAFE_FUNCTION=0
__INTERNAL__DEBUG_LOG_CONST=0

def test(order):
    __INTERNAL__DEBUG_LOG=1
    return order

print 'This comes first'
print test('second')
print 'and I am third'
"""

print buildcode(code)
