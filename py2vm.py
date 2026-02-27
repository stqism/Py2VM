# Python 2.7 opcode table (dis.opname equivalent — hardcoded to avoid dis import)
_OPNAME_RAW = {
    0: 'STOP_CODE', 1: 'POP_TOP', 2: 'PUSH_NULL', 3: 'ROT_THREE', 4: 'DUP_TOP',
    5: 'ROT_FOUR', 9: 'NOP', 10: 'UNARY_POSITIVE', 11: 'UNARY_NEGATIVE',
    12: 'UNARY_NOT', 13: 'UNARY_CONVERT', 15: 'UNARY_INVERT', 19: 'BINARY_POWER',
    20: 'BINARY_MULTIPLY', 21: 'BINARY_DIVIDE', 22: 'BINARY_MODULO',
    23: 'BINARY_ADD', 24: 'BINARY_SUBTRACT', 25: 'BINARY_SUBSCR',
    26: 'BINARY_FLOOR_DIVIDE', 27: 'BINARY_TRUE_DIVIDE',
    28: 'INPLACE_FLOOR_DIVIDE', 29: 'INPLACE_TRUE_DIVIDE',
    30: 'SLICE+0', 31: 'SLICE+1', 32: 'SLICE+2', 33: 'SLICE+3',
    40: 'STORE_SLICE+0', 41: 'STORE_SLICE+1', 42: 'STORE_SLICE+2', 43: 'STORE_SLICE+3',
    50: 'DELETE_SLICE+0', 51: 'DELETE_SLICE+1', 52: 'DELETE_SLICE+2', 53: 'DELETE_SLICE+3',
    54: 'STORE_MAP', 55: 'INPLACE_ADD', 56: 'INPLACE_SUBTRACT',
    57: 'INPLACE_MULTIPLY', 58: 'INPLACE_DIVIDE', 59: 'INPLACE_MODULO',
    60: 'STORE_SUBSCR', 61: 'DELETE_SUBSCR', 62: 'BINARY_LSHIFT',
    63: 'BINARY_RSHIFT', 64: 'BINARY_AND', 65: 'BINARY_XOR', 66: 'BINARY_OR',
    67: 'INPLACE_POWER', 68: 'GET_ITER', 70: 'PRINT_EXPR', 71: 'PRINT_ITEM',
    72: 'PRINT_NEWLINE', 73: 'PRINT_ITEM_TO', 74: 'PRINT_NEWLINE_TO',
    75: 'INPLACE_LSHIFT', 76: 'INPLACE_RSHIFT', 77: 'INPLACE_AND',
    78: 'INPLACE_XOR', 79: 'INPLACE_OR', 80: 'BREAK_LOOP', 81: 'WITH_CLEANUP',
    82: 'LOAD_LOCALS', 83: 'RETURN_VALUE', 84: 'IMPORT_STAR', 85: 'EXEC_STMT',
    86: 'YIELD_VALUE', 87: 'POP_BLOCK', 88: 'END_FINALLY', 89: 'BUILD_CLASS',
    90: 'STORE_NAME', 91: 'DELETE_NAME', 92: 'UNPACK_SEQUENCE', 93: 'FOR_ITER',
    94: 'LIST_APPEND', 95: 'STORE_ATTR', 96: 'DELETE_ATTR', 97: 'STORE_GLOBAL',
    98: 'DELETE_GLOBAL', 99: 'DUP_TOPX', 100: 'LOAD_CONST', 101: 'LOAD_NAME',
    102: 'BUILD_TUPLE', 103: 'BUILD_LIST', 104: 'BUILD_SET', 105: 'BUILD_MAP',
    106: 'LOAD_ATTR', 107: 'COMPARE_OP', 108: 'IMPORT_NAME', 109: 'IMPORT_FROM',
    110: 'JUMP_FORWARD', 111: 'JUMP_IF_FALSE_OR_POP', 112: 'JUMP_IF_TRUE_OR_POP',
    113: 'JUMP_ABSOLUTE', 114: 'POP_JUMP_IF_FALSE', 115: 'POP_JUMP_IF_TRUE',
    116: 'LOAD_GLOBAL', 119: 'CONTINUE_LOOP', 120: 'SETUP_LOOP',
    121: 'SETUP_EXCEPT', 122: 'SETUP_FINALLY', 124: 'LOAD_FAST',
    125: 'STORE_FAST', 126: 'DELETE_FAST', 130: 'RAISE_VARARGS',
    131: 'CALL_FUNCTION', 132: 'MAKE_FUNCTION', 133: 'BUILD_SLICE',
    134: 'MAKE_CLOSURE', 135: 'LOAD_CLOSURE', 136: 'LOAD_DEREF',
    137: 'STORE_DEREF', 140: 'CALL_FUNCTION_VAR', 141: 'CALL_FUNCTION_KW',
    142: 'CALL_FUNCTION_VAR_KW', 143: 'SETUP_WITH', 145: 'EXTENDED_ARG',
    146: 'SET_ADD', 147: 'MAP_ADD',
    # Python 3.11 wordcode additions / conflict overrides
    2: 'PUSH_NULL',                    # was ROT_TWO
    71: 'LOAD_BUILD_CLASS',            # was PRINT_ITEM
    114: 'POP_JUMP_FORWARD_IF_FALSE',  # was POP_JUMP_IF_FALSE
    115: 'POP_JUMP_FORWARD_IF_TRUE',   # was POP_JUMP_IF_TRUE
    122: 'BINARY_OP',                  # was SETUP_FINALLY
    129: 'POP_JUMP_FORWARD_IF_NONE',   # new
    140: 'JUMP_BACKWARD',              # was CALL_FUNCTION_VAR
    144: 'EXTENDED_ARG',               # was not in table (145 was in Python 2)
    151: 'RESUME', 160: 'LOAD_METHOD',
    164: 'DICT_MERGE', 165: 'DICT_UPDATE', 166: 'PRECALL', 171: 'CALL',
    175: 'POP_JUMP_BACKWARD_IF_FALSE', 176: 'POP_JUMP_BACKWARD_IF_TRUE',
    156: 'BUILD_CONST_KEY_MAP', 128: 'POP_JUMP_FORWARD_IF_NOT_NONE',
}
# Build OPNAME list without xrange — while loop with arithmetic only
_i = 0
OPNAME = []
while _i < 256:
    OPNAME.append(_OPNAME_RAW.get(_i, '<%d>' % _i))
    _i += 1
del _i

# Python 2.7 comparison operator table (dis.cmp_op equivalent)
CMP_OP = ('<', '<=', '==', '!=', '>', '>=', 'in', 'not in', 'is', 'is not',
          'exception match', 'BAD')


class _StringIO(object):
    """Minimal string buffer replacing the StringIO module."""
    def __init__(self):
        self._buf = []

    def write(self, s):
        self._buf.append(s)

    def getvalue(self):
        return ''.join(self._buf)


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
    raw = bytecode.co_code
    # Normalize co_code to a list of ints once.  In Python 3, bytes[i] already
    # yields an int.  In Python 2, co_code is a str and each element is a char
    # that needs ord() — there is no dunder equivalent for that conversion.
    code = []
    for b in raw:
        if b.__class__.__name__ == 'int':
            code.append(b)
        else:
            code.append(ord(b))  # Python 2: char -> int, no dunder alternative
    # Python 3 uses WORDCODE: every instruction is exactly 2 bytes (opcode, arg).
    # Python 2 uses variable-width: 1 byte for no-arg, 3 bytes for arg instructions.
    # Detect format by checking if co_code is a bytes object (Python 3) or str (Python 2).
    py3_wordcode = (raw.__class__.__name__ == 'bytes')
    i = 0
    index_counter = 0
    extended_arg = 0  # accumulator for EXTENDED_ARG prefix in wordcode mode
    while True:
        try:
            opcode_byte = code[i]
        except IndexError:
            break
        offset = i
        i += 1
        if py3_wordcode:
            # Python 3 WORDCODE: always read one arg byte
            try:
                arg = code[i]
            except IndexError:
                arg = 0
            i += 1
            if opcode_byte == 0:  # CACHE / padding — skip
                continue
            opcode_value = OPNAME[opcode_byte]
            if opcode_value == 'EXTENDED_ARG':
                # Accumulate high bits for the next instruction's arg
                extended_arg = (extended_arg | arg) << 8
                continue
            arg = extended_arg | arg
            extended_arg = 0
            # In wordcode, jump args are instruction counts (words), not bytes.
            # Multiply by 2 to convert to byte offsets.
            if opcode_value in ('JUMP_FORWARD', 'FOR_ITER',
                                'POP_JUMP_FORWARD_IF_FALSE',
                                'POP_JUMP_FORWARD_IF_TRUE',
                                'POP_JUMP_FORWARD_IF_NONE',
                                'POP_JUMP_FORWARD_IF_NOT_NONE',
                                'SETUP_LOOP', 'SETUP_EXCEPT', 'SETUP_FINALLY'):
                arg = i + arg * 2
            elif opcode_value in ('JUMP_BACKWARD',
                                  'POP_JUMP_BACKWARD_IF_TRUE',
                                  'POP_JUMP_BACKWARD_IF_FALSE'):
                arg = i - arg * 2
        else:
            # Python 2 variable-width format
            if opcode_byte == 0:
                continue
            opcode_value = OPNAME[opcode_byte]
            if hasarg(opcode_value):
                arg = code[i] | (code[i + 1] << 8)
                i += 2
                if opcode_value in ('JUMP_FORWARD', 'FOR_ITER',
                                    'SETUP_LOOP', 'SETUP_EXCEPT', 'SETUP_FINALLY'):
                    arg = i + arg
            else:
                arg = 0
        offset_to_index[offset] = index_counter
        bytecode_list.append([opcode_value, arg])
        index_counter += 1
    return bytecode_list, offset_to_index


def buildcode(code):
    return py2vm(compile(code, '<none>', 'exec'))


def py2vm(bytecode, stack=False, rec_log=False, fast_locals=None):

    if rec_log != False:
        log = rec_log
    else:
        log = _StringIO()
        log.write('py2vm output:\n')

    if stack != False:
        const_stack = stack
    else:
        const_stack = []

    fast_dict = fast_locals if fast_locals is not None else {}
    # Detect Python 3 wordcode: co_code is bytes (int-indexable) vs str (Python 2)
    py3_mode = (bytecode.co_code.__class__.__name__ == 'bytes')

    block_stack = []

    __INTERNAL__DEBUG_LOG = 1
    __INTERNAL__DEBUG_LOG_CONST = 0
    __INTERNAL__DEBUG_LOG_VAR = 0
    name_dict = {}

    # I was inspired by str8C
    i = -1
    opcode_array, offset_to_index = bytecode_optimize(bytecode)
    while True:
        i += 1
        try:
            opcode_pack = opcode_array[i]
        except IndexError:
            break
        opcode_value = opcode_pack[0]
        arg = opcode_pack[1]

        # log.write(const_stack)
        # log.write(opcode_value)

        if __INTERNAL__DEBUG_LOG_CONST:
            log.write(const_stack.__str__() + '\n')

        if __INTERNAL__DEBUG_LOG_VAR:
            log.write(name_dict.__str__() + '\n')

        if opcode_value == 'NOP':
            if __INTERNAL__DEBUG_LOG:
                log.write('DEBUG => internal placeholder\n')

        elif opcode_value == 'RESUME':
            # Python 3.11: entry point marker — treated as NOP
            if __INTERNAL__DEBUG_LOG:
                log.write('DEBUG => RESUME (Python 3.11 entry marker)\n')

        elif opcode_value == 'PUSH_NULL':
            # Python 3.11: push NULL sentinel before a non-method callable
            const_stack.insert(0, None)
            if __INTERNAL__DEBUG_LOG:
                log.write('DEBUG => PUSH_NULL\n')

        elif opcode_value == 'PRECALL':
            # Python 3.11: pre-call check — treated as NOP
            if __INTERNAL__DEBUG_LOG:
                log.write('DEBUG => PRECALL\n')

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
                    'DEBUG => loaded %s on to stack\n' % (bytecode.co_consts[arg],))

        elif opcode_value == 'LOAD_FAST':
            var_name = bytecode.co_varnames[arg]
            const_stack.insert(0, fast_dict.get(var_name))

            if __INTERNAL__DEBUG_LOG:
                log.write(
                    'DEBUG => loaded %s (%s) on to stack\n' % (var_name, fast_dict.get(var_name)))

        elif opcode_value == 'LOAD_NAME':
            _load_name_val = name_dict.get(arg)
            if _load_name_val is None:
                # Fall back to Python builtins for names like 'print', 'range', etc.
                _name_str = bytecode.co_names[arg]
                try:
                    _load_name_val = __builtins__[_name_str]
                except TypeError:
                    try:
                        _load_name_val = __builtins__.__dict__[_name_str]
                    except KeyError:
                        _load_name_val = None
                except KeyError:
                    _load_name_val = None
            const_stack.insert(0, _load_name_val)

            if __INTERNAL__DEBUG_LOG:
                log.write('DEBUG => loaded %s on to stack\n' % (_load_name_val,))

        elif opcode_value == 'LOAD_GLOBAL':
            # Python 3.11: arg = (name_index << 1) | push_null_flag
            # Python 2: arg is the name index directly
            _lg_idx = (arg >> 1) if py3_mode else arg
            _lg_push_null = (arg & 1) if py3_mode else 0
            _lg_name = bytecode.co_names[_lg_idx]
            # Check local name_dict first (keyed by index in current co_names)
            _lg_val = name_dict.get(_lg_idx)
            if _lg_val is None:
                try:
                    _lg_val = __builtins__[_lg_name]
                except TypeError:
                    try:
                        _lg_val = __builtins__.__dict__[_lg_name]
                    except KeyError:
                        _lg_val = None
                except KeyError:
                    _lg_val = None
            if _lg_push_null:
                const_stack.insert(0, None)
            const_stack.insert(0, _lg_val)

            if __INTERNAL__DEBUG_LOG:
                log.write('DEBUG => LOAD_GLOBAL %s => %s\n' % (_lg_name, _lg_val))

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
            const_stack.insert(0, const_stack[0].__repr__())

            if __INTERNAL__DEBUG_LOG:
                log.write("DEBUG => added repr(tos) to top of stack\n")

        elif opcode_value == 'UNARY_INVERT':
            const_stack.insert(0, ~const_stack[0])

            if __INTERNAL__DEBUG_LOG:
                log.write("DEBUG => added ~tos to top of stack\n")

        elif opcode_value == 'GET_ITER':
            const_stack[0] = const_stack[0].__iter__()

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
            math0 = const_stack.pop(0).__int__()
            math1 = const_stack.pop(0).__int__()
            const_stack.insert(0, math1 << math0)

            if __INTERNAL__DEBUG_LOG:
                log.write("DEBUG => shifted %s left %s\n" % (math1, math0))

        elif opcode_value == 'BINARY_RSHIFT':
            math0 = const_stack.pop(0).__int__()
            math1 = const_stack.pop(0).__int__()
            const_stack.insert(0, math1 >> math0)

            if __INTERNAL__DEBUG_LOG:
                log.write("DEBUG => shifted %s right %s\n" % (math1, math0))

        elif opcode_value == 'BINARY_AND':
            math0 = const_stack.pop(0).__int__()
            math1 = const_stack.pop(0).__int__()
            const_stack.insert(0, math1 & math0)

            if __INTERNAL__DEBUG_LOG:
                log.write("DEBUG => %s AND %s\n" % (math1, math0))

        elif opcode_value == 'BINARY_XOR':
            math0 = const_stack.pop(0).__int__()
            math1 = const_stack.pop(0).__int__()
            const_stack.insert(0, math1 ^ math0)

            if __INTERNAL__DEBUG_LOG:
                log.write("DEBUG => %s XOR %s\n" % (math1, math0))

        elif opcode_value == 'BINARY_OR':
            math0 = const_stack.pop(0).__int__()
            math1 = const_stack.pop(0).__int__()
            const_stack.insert(0, math1 | math0)

            if __INTERNAL__DEBUG_LOG:
                log.write("DEBUG => %s OR %s\n" % (math1, math0))

        elif opcode_value == 'BINARY_OP':
            # Python 3.11: unified binary/inplace op; arg selects operation.
            # Regular: 0=+  1=&  2=//  3=<<  5=*  6=%  7=|  8=**  9=>>  10=-  11=/  12=^
            # Inplace: add 13 to regular arg (13=+=, 14=&=, ..., 23=-=, etc.)
            _bo_arg = arg if arg < 13 else arg - 13
            math0 = const_stack.pop(0)
            math1 = const_stack.pop(0)
            if _bo_arg == 0:    result = math1 + math0
            elif _bo_arg == 1:  result = math1 & math0
            elif _bo_arg == 2:  result = math1 // math0
            elif _bo_arg == 3:  result = math1 << math0
            elif _bo_arg == 5:  result = math1 * math0
            elif _bo_arg == 6:  result = math1 % math0
            elif _bo_arg == 7:  result = math1 | math0
            elif _bo_arg == 8:  result = math1 ** math0
            elif _bo_arg == 9:  result = math1 >> math0
            elif _bo_arg == 10: result = math1 - math0
            elif _bo_arg == 11: result = math1 / math0
            elif _bo_arg == 12: result = math1 ^ math0
            else:               result = None
            const_stack.insert(0, result)

            if __INTERNAL__DEBUG_LOG:
                log.write("DEBUG => BINARY_OP[%d] %s op %s => %s\n" % (arg, math1, math0, result))

        elif opcode_value == 'LOAD_BUILD_CLASS':
            # Python 3.11: push __build_class__ builtin for class definitions
            try:
                _bld = __builtins__['__build_class__']
            except TypeError:
                _bld = __builtins__.__build_class__
            const_stack.insert(0, _bld)

            if __INTERNAL__DEBUG_LOG:
                log.write("DEBUG => LOAD_BUILD_CLASS\n")

        elif opcode_value == 'COMPARE_OP':
            math0 = const_stack.pop(0)
            math1 = const_stack.pop(0)
            op_name = CMP_OP[arg]
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

        elif opcode_value == 'JUMP_BACKWARD':
            # Python 3.11: backward loop jump (replaces JUMP_ABSOLUTE for loops)
            i = offset_to_index[arg] - 1

            if __INTERNAL__DEBUG_LOG:
                log.write("DEBUG => jumped backward to %s\n" % (arg))

        elif opcode_value == 'POP_JUMP_FORWARD_IF_FALSE':
            # Python 3.11: forward conditional jump if false
            tos = const_stack.pop(0)
            if not tos:
                i = offset_to_index[arg] - 1

            if __INTERNAL__DEBUG_LOG:
                log.write("DEBUG => POP_JUMP_FORWARD_IF_FALSE: tos=%s target=%s\n" % (tos, arg))

        elif opcode_value == 'POP_JUMP_FORWARD_IF_TRUE':
            # Python 3.11: forward conditional jump if true
            tos = const_stack.pop(0)
            if tos:
                i = offset_to_index[arg] - 1

            if __INTERNAL__DEBUG_LOG:
                log.write("DEBUG => POP_JUMP_FORWARD_IF_TRUE: tos=%s target=%s\n" % (tos, arg))

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
                try:
                    val = const_stack[0].__next__()  # Python 3
                except AttributeError:
                    val = const_stack[0].next()  # Python 2
                const_stack.insert(0, val)
            except StopIteration:
                const_stack.pop(0)
                i = offset_to_index[arg] - 1

            if __INTERNAL__DEBUG_LOG:
                log.write("DEBUG => FOR_ITER\n")

        elif opcode_value == 'PRINT_ITEM':
            log.write(const_stack.pop(0).__str__())

        elif opcode_value == 'PRINT_NEWLINE':
            log.write('\n')

        elif opcode_value == 'LOAD_METHOD':
            # Python 3.11: load a method; push (attr, obj) so CALL can pop self
            _lm_obj = const_stack.pop(0)
            _lm_attr = _lm_obj.__getattribute__(bytecode.co_names[arg])
            const_stack.insert(0, _lm_attr)  # method at TOS
            const_stack.insert(1, _lm_obj)   # self below method (CALL will pop it)

            if __INTERNAL__DEBUG_LOG:
                log.write("DEBUG => LOAD_METHOD %s\n" % bytecode.co_names[arg])

        elif opcode_value == 'BUILD_MAP':
            # arg=0: push empty dict (size hint).
            # arg>0: pop 2*arg items (val0, key0, val1, key1, ...) and build dict.
            if arg == 0:
                const_stack.insert(0, {})
            else:
                _bm_dict = {}
                _bm_count = arg
                while _bm_count > 0:
                    _bm_val = const_stack.pop(0)
                    _bm_key = const_stack.pop(0)
                    _bm_dict[_bm_key] = _bm_val
                    _bm_count -= 1
                const_stack.insert(0, _bm_dict)

            if __INTERNAL__DEBUG_LOG:
                log.write("DEBUG => BUILD_MAP %d\n" % arg)

        elif opcode_value == 'MAP_ADD':
            # MAP_ADD i: pop TOS (value) and TOS1 (key), add to dict at stack[i-1]
            _ma_value = const_stack.pop(0)
            _ma_key = const_stack.pop(0)
            const_stack[arg - 1][_ma_key] = _ma_value

        elif opcode_value == 'BUILD_CONST_KEY_MAP':
            # Stack: TOS=keys_tuple, then N values (arg=N). Build dict.
            _bck_keys = const_stack.pop(0)
            _bck_vals = []
            _bck_count = arg
            while _bck_count > 0:
                _bck_vals.insert(0, const_stack.pop(0))
                _bck_count -= 1
            _bck_result = {}
            _bck_idx = 0
            for _bck_k in _bck_keys:
                _bck_result[_bck_k] = _bck_vals[_bck_idx]
                _bck_idx += 1
            const_stack.insert(0, _bck_result)

            if __INTERNAL__DEBUG_LOG:
                log.write("DEBUG => BUILD_CONST_KEY_MAP %d\n" % arg)

        elif opcode_value in ('DICT_UPDATE', 'DICT_MERGE'):
            # DICT_UPDATE/DICT_MERGE i: pop TOS, merge into dict at stack[i-1]
            _du_other = const_stack.pop(0)
            const_stack[arg - 1].update(_du_other)

            if __INTERNAL__DEBUG_LOG:
                log.write("DEBUG => %s\n" % opcode_value)

        elif opcode_value == 'BUILD_LIST':
            # arg items from stack (0 = empty list)
            _bl_items = []
            _bl_count = arg
            while _bl_count > 0:
                _bl_items.insert(0, const_stack.pop(0))
                _bl_count -= 1
            const_stack.insert(0, _bl_items)

            if __INTERNAL__DEBUG_LOG:
                log.write("DEBUG => BUILD_LIST %d\n" % arg)

        elif opcode_value == 'BUILD_TUPLE':
            # arg items from stack (0 = empty tuple)
            _bt_items = []
            _bt_count = arg
            while _bt_count > 0:
                _bt_items.insert(0, const_stack.pop(0))
                _bt_count -= 1
            const_stack.insert(0, tuple(_bt_items))

            if __INTERNAL__DEBUG_LOG:
                log.write("DEBUG => BUILD_TUPLE %d\n" % arg)

        elif opcode_value == 'POP_JUMP_BACKWARD_IF_TRUE':
            tos = const_stack.pop(0)
            if tos:
                i = offset_to_index[arg] - 1

            if __INTERNAL__DEBUG_LOG:
                log.write("DEBUG => POP_JUMP_BACKWARD_IF_TRUE: tos=%s target=%s\n" % (tos, arg))

        elif opcode_value == 'POP_JUMP_BACKWARD_IF_FALSE':
            tos = const_stack.pop(0)
            if not tos:
                i = offset_to_index[arg] - 1

            if __INTERNAL__DEBUG_LOG:
                log.write("DEBUG => POP_JUMP_BACKWARD_IF_FALSE: tos=%s target=%s\n" % (tos, arg))

        elif opcode_value == 'POP_JUMP_FORWARD_IF_NONE':
            tos = const_stack.pop(0)
            if tos is None:
                i = offset_to_index[arg] - 1

            if __INTERNAL__DEBUG_LOG:
                log.write("DEBUG => POP_JUMP_FORWARD_IF_NONE: tos=%s target=%s\n" % (tos, arg))

        elif opcode_value == 'POP_JUMP_FORWARD_IF_NOT_NONE':
            tos = const_stack.pop(0)
            if tos is not None:
                i = offset_to_index[arg] - 1

            if __INTERNAL__DEBUG_LOG:
                log.write("DEBUG => POP_JUMP_FORWARD_IF_NOT_NONE: tos=%s target=%s\n" % (tos, arg))

        elif opcode_value in ('PUSH_EXC_INFO', 'POP_EXCEPT', 'CHECK_EXC_MATCH',
                              'COPY', 'RERAISE'):
            # Minimal exception-handling stubs — just discard the exception data
            # so execution can continue on the happy path.
            if __INTERNAL__DEBUG_LOG:
                log.write("DEBUG => exc-handling stub: %s\n" % opcode_value)

        elif opcode_value == 'LOAD_ATTR':
            math0 = const_stack.pop(0)
            const_stack.insert(0, math0.__getattribute__(bytecode.co_names[arg]))

        elif opcode_value == 'MAKE_FUNCTION':
            pass  # code object stays on TOS; STORE_NAME will consume it

        elif opcode_value == 'CALL_FUNCTION':
            argc = arg & 0xff
            kwargc = (arg >> 8) & 0xff

            # Collect keyword arguments — while countdown avoids xrange/range
            kwargs = {}
            count = kwargc
            while count > 0:
                val = const_stack.pop(0)
                key = const_stack.pop(0)
                kwargs[key] = val
                count -= 1

            # Collect positional arguments in call order
            args_list = []
            count = argc
            while count > 0:
                args_list.insert(0, const_stack.pop(0))
                count -= 1

            func = const_stack.pop(0)

            # Dispatch: probe for co_code to detect VM-interpreted code objects;
            # fall back to native callable otherwise.  try/except replaces hasattr.
            try:
                func.co_code  # probe — only code objects have this attribute
                fl = {}
                idx = 0
                for val in args_list:
                    try:
                        fl[func.co_varnames[idx]] = val
                    except IndexError:
                        pass
                    idx += 1
                result_stack, log = py2vm(func, [], log, fast_locals=fl)
                const_stack.insert(0, result_stack[0] if result_stack else None)

                if __INTERNAL__DEBUG_LOG:
                    log.write("DEBUG => called function %s(%s)\n" % (func.co_name, args_list))

            except AttributeError:
                try:
                    result = func(*args_list, **kwargs)
                    const_stack.insert(0, result)
                except Exception as e:
                    log.write("ERROR => could not call %s: %s\n" % (func, e))
                    const_stack.insert(0, None)

        elif opcode_value == 'CALL':
            # Python 3.11 CALL opcode: like CALL_FUNCTION but stack also has a
            # NULL sentinel below the callable (pushed by PUSH_NULL).
            argc = arg  # no kwargs encoding in Python 3.11 CALL

            args_list = []
            count = argc
            while count > 0:
                args_list.insert(0, const_stack.pop(0))
                count -= 1

            func = const_stack.pop(0)
            # Pop the NULL sentinel pushed by PUSH_NULL (or self for method calls)
            if const_stack:
                const_stack.pop(0)

            try:
                func.co_code  # probe for VM code object
                fl = {}
                idx = 0
                for val in args_list:
                    try:
                        fl[func.co_varnames[idx]] = val
                    except IndexError:
                        pass
                    idx += 1
                result_stack, log = py2vm(func, [], log, fast_locals=fl)
                const_stack.insert(0, result_stack[0] if result_stack else None)

                if __INTERNAL__DEBUG_LOG:
                    log.write("DEBUG => CALL code object %s(%s)\n" % (func.co_name, args_list))

            except AttributeError:
                try:
                    result = func(*args_list)
                    const_stack.insert(0, result)
                except Exception as e:
                    log.write("ERROR => could not CALL %s: %s\n" % (func, e))
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


def run_script(path):
    """Read a Python source file and execute it through the VM."""
    _f = open(path)
    _src = _f.read()
    _f.close()
    return buildcode(_src)


_argv = __import__('sys').argv
try:
    _script = _argv[1]
except IndexError:
    _script = None

if _script is not None:
    print(run_script(_script))
else:
    code = """
__INTERNAL__DEBUG_LOG=1
__INTERNAL__DEBUG_LOG_CONST=0

def test(order):
    __INTERNAL__DEBUG_LOG=1
    return order

print('This comes first')
print(test('second'))
print('and I am third')
"""
    print(buildcode(code))
