from itertools import tee, islice, chain, izip
from ctypes import *
import dis
import binascii
import sys
import StringIO


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


def item_arg(some_iterable):
    item, arg = tee(some_iterable, 2)
    arg = chain(islice(arg, 1, None), [None])
    return izip(item, arg)


def bytecode_optimize(bytecode):
    bytecode_list = []
    skipme = False
    for opcode, arg in item_arg(bytecode.co_code):
        if skipme == True:
            skipme = False
            continue
        opcode_hex = int(binascii.b2a_hex(opcode), 16)
        # This shit is ignored, despite taking half the opcode
        if opcode_hex == 0x0:
            pass
        else:
            opcode_value = dis.opname[opcode_hex]
            if opcode_value == 'POP_BLOCK':
                opcode_value = 'POP_TOP'

            if hasarg(opcode_value):
                bytecode_list.append(
                    [opcode_value, int(binascii.b2a_hex(arg), 16)])
                bytecode_list.append(['ARG', 0])
                bytecode_list.append(['ARG', 0])
                skipme = True
            else:
                bytecode_list.append([opcode_value, 0])
    return bytecode_list


def py2vm(code):
    bytecode = compile(code, '<none>', 'exec')
    log = StringIO.StringIO()
    # print binascii.b2a_hex(bytecode.co_code)
    log.write(code)
    log.write('=>\n')
    #opcode = dis.dis(bytecode)
    # log.write(opcode)
    # log.write('=>\n')
    const_stack = []
    jump = 0
    libc = CDLL("/usr/lib/libc.dylib")

    __INTERNAL__DEBUG_LOG = 0
    __INTERNAL__DEBUG_LOG_CONST = 0
    __INTERNAL__DEBUG_LOG_VAR = 0
    name_dict = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 0: 0}

    # I was inspired by str8C
    i = -1
    opcode_array = bytecode_optimize(bytecode)
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

        elif opcode_value == 'ARG':
            pass

        elif opcode_value == 'POP_TOP':
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

        elif opcode_value == 'LOAD_NAME':
            const_stack.insert(0, name_dict[arg])

            if __INTERNAL__DEBUG_LOG:
                log.write('DEBUG => loaded %s on to stack\n' %
                          (name_dict[arg]))

        elif opcode_value == 'STORE_NAME':
            name_dict[arg] = const_stack[0]
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

            if __INTERNAL__DEBUG_LOG:
                log.write("DEBUG => set value of var %s to %s\n" %
                          (bytecode.co_names[arg], const_stack[0]))

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
            const_stack.insert(0, `const_stack[0]`)

            if __INTERNAL__DEBUG_LOG:
                log.write("DEBUG => added `tos` to top of stack\n")

        elif opcode_value == 'UNARY_INVERT':
            const_stack.insert(0, ~const_stack[0])

            if __INTERNAL__DEBUG_LOG:
                log.write("DEBUG => added ~tos to top of stack\n")

        elif opcode_value == 'GET_ITER':
            const_stack.insert(0, iter(const_stack[0]))

            if __INTERNAL__DEBUG_LOG:
                log.write("DEBUG => added iter(tos) to top of stack\n")

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
            if math0 == math1:
                const_stack.insert(0, 1)
            else:
                const_stack.insert(0, 0)

            if __INTERNAL__DEBUG_LOG:
                log.write("DEBUG => compared %s with %s\n" % (math0, math1))

        # TODO: proper stack jumping, watch byte counter
        elif opcode_value == 'POP_JUMP_IF_FALSE':
            if const_stack[0] == True:
                i = arg - 1

            if __INTERNAL__DEBUG_LOG:
                log.write("DEBUG => hit if, logic_outcome: %s\n" % (arg))

        elif opcode_value == 'POP_JUMP_IF_TRUE':
            if const_stack[0] == False:
                i = arg - 1
                del const_stack[0]
            if __INTERNAL__DEBUG_LOG:
                log.write("DEBUG => hit if, logic_outcome: %s\n" % (arg))

        elif opcode_value == 'JUMP_IF_TRUE_OR_POP':
            if const_stack[0] == False:
                i = arg - 1
            else:
                del const_stack[0]
            if __INTERNAL__DEBUG_LOG:
                log.write("DEBUG => hit if, logic_outcome: %s\n" % (arg))

        elif opcode_value == 'JUMP_IF_FALSE_OR_POP':
            if const_stack[0] == True:
                i = arg - 1
            else:
                del const_stack[0]
            if __INTERNAL__DEBUG_LOG:
                log.write("DEBUG => hit if, logic_outcome: %s\n" % (arg))

        elif opcode_value == 'JUMP_FORWARD':
            i += arg + 1
            if __INTERNAL__DEBUG_LOG:
                log.write("DEBUG => jumped forward %s\n" % (arg))

        elif opcode_value == 'JUMP_ABSOLUTE':
            i = arg - 1
            if __INTERNAL__DEBUG_LOG:
                log.write("DEBUG => jumped to %s\n" % (arg))

        elif opcode_value == 'SETUP_LOOP':
            # i = arg - 1
            if __INTERNAL__DEBUG_LOG:
                log.write("DEBUG => jumped to %s\n" % (arg))

        elif opcode_value == 'PRINT_ITEM':
            log.write(str(const_stack[0]))

        elif opcode_value == 'PRINT_NEWLINE':
            log.write('\n')

        elif opcode_value == 'LOAD_ATTR':
            math0 = const_stack.pop(0)
            const_stack.insert(0, getattr(math0, bytecode.co_names[arg]))

        elif opcode_value == 'CALL_FUNCTION':
            try:
                function = bytecode.co_names[const_stack[arg]]
            except:
                function = const_stack[arg]
            arguments = []

            for tick in xrange(0, arg):
                arguments.insert(0, const_stack[tick])

            if function == 'load.library':
                const_stack.insert(0, CDLL(arguments[0]))
                # print dir(const_stack[0])

            elif function == '__INTERNAL__libc':
                const_stack.insert(0, libc)

            else:
                try:
                    lookup = const_stack[1](const_stack[0])
                    const_stack.insert(0, lookup)
                except:
                    print const_stack

        elif opcode_value == 'RETURN_VALUE':

            if __INTERNAL__DEBUG_LOG:
                log.write("DEBUG => returned execution back to parent\n")
            break

        else:
            log.write("unknown opcode: %s\n" % (opcode_value))

    i += 1
    return log.getvalue()

code = """
__INTERNAL__DEBUG_LOG=1
__INTERNAL__DEBUG_LOG_CONST=0

print i
print i
print i

"""

print py2vm(code)
