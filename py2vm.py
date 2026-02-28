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
    35: 'PUSH_EXC_INFO',               # new in 3.11
    36: 'CHECK_EXC_MATCH',             # new in 3.11
    71: 'LOAD_BUILD_CLASS',            # was PRINT_ITEM
    89: 'POP_EXCEPT',                  # was BUILD_CLASS in Python 2
    99: 'SWAP',                        # was DUP_TOPX in Python 2
    114: 'POP_JUMP_FORWARD_IF_FALSE',  # was POP_JUMP_IF_FALSE
    115: 'POP_JUMP_FORWARD_IF_TRUE',   # was POP_JUMP_IF_TRUE
    117: 'IS_OP',                      # new in 3.9+
    118: 'CONTAINS_OP',               # new in 3.9+
    119: 'RERAISE',                    # was CONTINUE_LOOP in Python 2
    120: 'COPY',                       # was SETUP_LOOP in Python 2
    122: 'BINARY_OP',                  # was SETUP_FINALLY
    128: 'POP_JUMP_FORWARD_IF_NOT_NONE',  # new
    129: 'POP_JUMP_FORWARD_IF_NONE',   # new
    135: 'MAKE_CELL',                  # was LOAD_CLOSURE in Python 2; Python 3.11 shifted by 1
    136: 'LOAD_CLOSURE',               # was LOAD_DEREF in Python 2
    137: 'LOAD_DEREF',                 # was STORE_DEREF in Python 2
    138: 'STORE_DEREF',                # new slot in Python 3.11
    140: 'JUMP_BACKWARD',              # was CALL_FUNCTION_VAR
    142: 'CALL_FUNCTION_EX',           # was CALL_FUNCTION_VAR_KW in Python 2
    144: 'EXTENDED_ARG',               # was not in table (145 was in Python 2)
    149: 'COPY_FREE_VARS',             # new in 3.11
    151: 'RESUME', 155: 'FORMAT_VALUE', 156: 'BUILD_CONST_KEY_MAP',
    157: 'BUILD_STRING', 160: 'LOAD_METHOD',
    164: 'DICT_MERGE', 165: 'DICT_UPDATE', 166: 'PRECALL', 171: 'CALL',
    172: 'KW_NAMES',
    175: 'POP_JUMP_BACKWARD_IF_FALSE', 176: 'POP_JUMP_BACKWARD_IF_TRUE',
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
    pending_ext_offsets = []  # EXTENDED_ARG offsets waiting to be mapped
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
                # Accumulate high bits for the next instruction's arg.
                # Save the offset so jump targets that land here map to
                # the following real instruction.
                extended_arg = (extended_arg | arg) << 8
                pending_ext_offsets.append(offset)
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
        # Map any preceding EXTENDED_ARG offsets to this instruction's index
        # so that jump targets pointing to an EXTENDED_ARG site resolve correctly.
        for _ext_off in pending_ext_offsets:
            offset_to_index[_ext_off] = index_counter
        pending_ext_offsets = []
        offset_to_index[offset] = index_counter
        bytecode_list.append([opcode_value, arg, offset])
        index_counter += 1
    return bytecode_list, offset_to_index


def buildcode(code):
    return py2vm(compile(code, '<none>', 'exec'))


def py2vm(bytecode, stack=False, rec_log=False, fast_locals=None, globals_frame=None):

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
    globals_frame = globals_frame if globals_frame is not None else {'__builtins__': __builtins__}
    # Cell variable storage: maps localsplus-index → [value] (mutable cell)
    _cells = {}
    # Free variable cells: list of [value] cells, one per co_freevars entry
    _free_cells = [None] * len(bytecode.co_freevars)

    # Factory: create a real callable that runs a code object through the VM.
    # Using a factory rather than a bare closure ensures each call captures its
    # own _co/_gf values rather than sharing a reference to the loop variable.
    def _mf_make(_co, _gf, _defaults=(), _closure=None):
        def _mf_callable(*_args, **_kwargs):
            _fl = {}
            # Apply default values first so all parameters have correct defaults.
            # _defaults is the positional-defaults tuple captured at MAKE_FUNCTION
            # time (Python stores defaults on the function, not the code object).
            _num_params = _co.co_argcount
            _num_defaults = len(_defaults)
            _first_default = _num_params - _num_defaults
            _di = 0
            while _di < _num_defaults:
                _fl[_co.co_varnames[_first_default + _di]] = _defaults[_di]
                _di += 1
            # Apply positional args to regular parameters (indices 0..co_argcount-1).
            _idx = 0
            while _idx < _num_params and _idx < len(_args):
                _fl[_co.co_varnames[_idx]] = _args[_idx]
                _idx += 1
            # If the function has a *args parameter (CO_VARARGS = 0x04), assign the
            # remaining positional args as a tuple to co_varnames[co_argcount].
            if _co.co_flags & 0x04:
                _fl[_co.co_varnames[_num_params]] = _args[_num_params:]
            # If the function has a **kwargs parameter (CO_VARKEYWORDS = 0x08),
            # assign the entire kwargs dict to the **kwargs variable name.
            # Otherwise spread individual keyword args by name (handles VM-level
            # calls like py2vm(..., fast_locals=..., globals_frame=...)).
            if _co.co_flags & 0x08:
                _kw_idx = _num_params + (1 if _co.co_flags & 0x04 else 0)
                _fl[_co.co_varnames[_kw_idx]] = _kwargs
            else:
                for _kname, _kval in _kwargs.items():
                    _fl[_kname] = _kval
            # Pass closure cells so COPY_FREE_VARS can populate _free_cells.
            if _closure is not None:
                _fl['__closure__'] = _closure
            _rs, _ig = py2vm(_co, [], False, fast_locals=_fl, globals_frame=_gf)
            return _rs[0] if _rs else None
        return _mf_callable

    # I was inspired by str8C
    i = -1
    opcode_array, offset_to_index = bytecode_optimize(bytecode)

    # Build exception table for Python 3.11+ so try/except in interpreted code works.
    _exc_table = []
    if py3_mode and hasattr(bytecode, 'co_exceptiontable') and bytecode.co_exceptiontable:
        try:
            import dis as _dis_mod
            for _et in _dis_mod._parse_exception_table(bytecode):
                _exc_table.append((_et.start, _et.end, _et.target, _et.depth))
        except Exception:
            pass

    # Keyword-argument names set by KW_NAMES before a CALL opcode.
    _kw_names = ()

    while True:
        i += 1
        try:
            opcode_pack = opcode_array[i]
        except IndexError:
            break
        opcode_value = opcode_pack[0]
        arg = opcode_pack[1]
        # Bytecode offset of this instruction (used for exception table lookup).
        _cur_offset = opcode_pack[2] if len(opcode_pack) > 2 else -1

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
            # Check local name_dict first (keyed by index in current co_names),
            # then the shared string-keyed globals_frame, then builtins.
            _lg_val = name_dict.get(_lg_idx)
            if _lg_val is None:
                _lg_val = globals_frame.get(_lg_name)
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

        elif opcode_value == 'MAKE_CELL':
            # Python 3.11: convert local at localsplus[arg] into a cell object.
            # arg is the localsplus index; for cellvars that are also params,
            # this equals the co_varnames index.
            _mc_name = bytecode.co_varnames[arg] if arg < len(bytecode.co_varnames) else ''
            _cells[arg] = [fast_dict.get(_mc_name)]
            if __INTERNAL__DEBUG_LOG:
                log.write("DEBUG => MAKE_CELL %d (%s)\n" % (arg, _mc_name))

        elif opcode_value == 'LOAD_CLOSURE':
            # Push the cell object at localsplus[arg] (a cellvar).
            const_stack.insert(0, _cells.get(arg, [None]))
            if __INTERNAL__DEBUG_LOG:
                _lc_name = bytecode.co_varnames[arg] if arg < len(bytecode.co_varnames) else str(arg)
                log.write("DEBUG => LOAD_CLOSURE %d (%s)\n" % (arg, _lc_name))

        elif opcode_value == 'LOAD_DEREF':
            # arg < len(co_varnames): cellvar at _cells[arg]; else freevar.
            _ld_n = len(bytecode.co_varnames)
            if arg < _ld_n:
                _ld_cell = _cells.get(arg, [None])
            else:
                _ld_fi = arg - _ld_n
                _ld_cell = _free_cells[_ld_fi] if _ld_fi < len(_free_cells) else [None]
            const_stack.insert(0, _ld_cell[0] if _ld_cell is not None else None)
            if __INTERNAL__DEBUG_LOG:
                log.write("DEBUG => LOAD_DEREF %d => %r\n" % (arg, const_stack[0]))

        elif opcode_value == 'STORE_DEREF':
            _sd_val = const_stack.pop(0)
            _sd_n = len(bytecode.co_varnames)
            if arg < _sd_n:
                if arg not in _cells:
                    _cells[arg] = [None]
                _cells[arg][0] = _sd_val
            else:
                _sd_fi = arg - _sd_n
                if _sd_fi < len(_free_cells) and _free_cells[_sd_fi] is not None:
                    _free_cells[_sd_fi][0] = _sd_val
            if __INTERNAL__DEBUG_LOG:
                log.write("DEBUG => STORE_DEREF %d\n" % arg)

        elif opcode_value == 'COPY_FREE_VARS':
            # Populate _free_cells[0..arg-1] from fast_dict['__closure__'].
            _cfv_closure = fast_dict.get('__closure__')
            if _cfv_closure is not None:
                _cfv_i = 0
                while _cfv_i < arg and _cfv_i < len(_cfv_closure):
                    _free_cells[_cfv_i] = _cfv_closure[_cfv_i]
                    _cfv_i += 1
            if __INTERNAL__DEBUG_LOG:
                log.write("DEBUG => COPY_FREE_VARS %d\n" % arg)

        elif opcode_value == 'SWAP':
            # SWAP(i): swap TOS (index 0) with stack[i-1] (1-indexed from TOS).
            if arg > 1 and len(const_stack) >= arg:
                const_stack[0], const_stack[arg - 1] = const_stack[arg - 1], const_stack[0]
            if __INTERNAL__DEBUG_LOG:
                log.write("DEBUG => SWAP %d\n" % arg)

        elif opcode_value == 'STORE_NAME':
            name_dict[arg] = const_stack.pop(0)
            globals_frame[bytecode.co_names[arg]] = name_dict[arg]
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

        elif opcode_value == 'STORE_GLOBAL':
            globals_frame[bytecode.co_names[arg]] = const_stack.pop(0)

            if __INTERNAL__DEBUG_LOG:
                log.write("DEBUG => STORE_GLOBAL %s\n" % bytecode.co_names[arg])

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
            try:
                const_stack.insert(0, math1[math0])
                if __INTERNAL__DEBUG_LOG:
                    log.write("DEBUG => set tos to %s[%s]\n" % (math1, math0))
            except Exception as _subscr_exc:
                _subscr_handled = False
                for _et_s, _et_e, _et_t, _et_d in _exc_table:
                    if _et_s <= _cur_offset < _et_e:
                        while len(const_stack) > _et_d:
                            const_stack.pop(0)
                        const_stack.insert(0, _subscr_exc)
                        i = offset_to_index.get(_et_t, i) - 1
                        _subscr_handled = True
                        break
                if not _subscr_handled:
                    log.write("ERROR => BINARY_SUBSCR failed: %s\n" % _subscr_exc)
                    const_stack.insert(0, None)

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
            # DICT_UPDATE/DICT_MERGE i: pop TOS (source), merge into dict at stack[i-1]
            _du_other = const_stack.pop(0)
            _du_dst = const_stack[arg - 1] if len(const_stack) >= arg else None
            if isinstance(_du_dst, dict):
                _du_dst.update(_du_other)
            else:
                log.write("ERROR => %s: dst at [%d] is %r, src=%r, stack depth=%d\n" % (
                    opcode_value, arg - 1, _du_dst, _du_other, len(const_stack)))

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

        elif opcode_value == 'UNPACK_SEQUENCE':
            # Pop TOS sequence, push its items right-to-left so TOS is item 0.
            _us_seq = list(const_stack.pop(0))
            _us_i = len(_us_seq) - 1
            while _us_i >= 0:
                const_stack.insert(0, _us_seq[_us_i])
                _us_i -= 1

            if __INTERNAL__DEBUG_LOG:
                log.write("DEBUG => UNPACK_SEQUENCE %d\n" % arg)

        elif opcode_value == 'IS_OP':
            # IS_OP(invert): TOS1 is TOS → push bool; invert=1 means "is not".
            _io_b = const_stack.pop(0)
            _io_a = const_stack.pop(0)
            _io_result = (_io_a is not _io_b) if arg else (_io_a is _io_b)
            const_stack.insert(0, _io_result)

        elif opcode_value == 'CONTAINS_OP':
            # CONTAINS_OP(invert): TOS1 in TOS → push bool; invert=1 means "not in".
            _co_container = const_stack.pop(0)
            _co_item = const_stack.pop(0)
            try:
                _co_result = (_co_item not in _co_container) if arg else (_co_item in _co_container)
            except TypeError:
                _co_result = False
            const_stack.insert(0, _co_result)

        elif opcode_value == 'STORE_SUBSCR':
            # TOS1[TOS] = TOS2; pops all three.
            _ss_key = const_stack.pop(0)
            _ss_obj = const_stack.pop(0)
            _ss_val = const_stack.pop(0)
            _ss_obj[_ss_key] = _ss_val

        elif opcode_value == 'DELETE_SUBSCR':
            # del TOS1[TOS]; pops both.
            _ds_key = const_stack.pop(0)
            _ds_obj = const_stack.pop(0)
            del _ds_obj[_ds_key]

        elif opcode_value == 'DELETE_FAST':
            _df_name = bytecode.co_varnames[arg]
            fast_dict.pop(_df_name, None)

        elif opcode_value == 'KW_NAMES':
            # Store keyword argument names for the next CALL opcode.
            _kw_names = bytecode.co_consts[arg]

        elif opcode_value == 'FORMAT_VALUE':
            # f-string value slot: arg encodes conversion + whether format spec present.
            _fv_have_spec = bool(arg & 0x04)
            _fv_conv = arg & 0x03
            _fv_spec = const_stack.pop(0) if _fv_have_spec else ''
            _fv_val = const_stack.pop(0)
            if _fv_conv == 1:
                _fv_val = str(_fv_val)
            elif _fv_conv == 2:
                _fv_val = repr(_fv_val)
            elif _fv_conv == 3:
                _fv_val = ascii(_fv_val)
            const_stack.insert(0, format(_fv_val, _fv_spec))

        elif opcode_value == 'BUILD_STRING':
            # Concatenate arg strings from the stack.
            _bstr_parts = []
            _bstr_n = arg
            while _bstr_n > 0:
                _bstr_parts.insert(0, const_stack.pop(0))
                _bstr_n -= 1
            const_stack.insert(0, ''.join(_bstr_parts))

        elif opcode_value == 'CALL_FUNCTION_EX':
            # CALL_FUNCTION_EX(flags): call func(*args[, **kwargs]).
            _cfex_kwargs = const_stack.pop(0) if (arg & 0x01) else {}
            _cfex_args = const_stack.pop(0)
            _cfex_func = const_stack.pop(0)
            if const_stack and const_stack[0] is None:
                const_stack.pop(0)  # pop NULL sentinel
            try:
                const_stack.insert(0, _cfex_func(*_cfex_args, **_cfex_kwargs))
            except Exception as _cfex_exc:
                log.write("ERROR => CALL_FUNCTION_EX failed: %s\n" % _cfex_exc)
                const_stack.insert(0, None)


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

        elif opcode_value == 'PUSH_EXC_INFO':
            # Pushes the caught exception; saves previous exception state (we use None).
            _pei_exc = const_stack.pop(0)
            const_stack.insert(0, None)      # placeholder for previous exception
            const_stack.insert(0, _pei_exc)  # caught exception back at TOS
            if __INTERNAL__DEBUG_LOG:
                log.write("DEBUG => PUSH_EXC_INFO\n")

        elif opcode_value == 'CHECK_EXC_MATCH':
            # Pops TOS (exception type), checks TOS1 (exception), pushes bool.
            _cem_types = const_stack.pop(0)
            _cem_exc = const_stack[0]  # stays on stack
            try:
                _cem_result = isinstance(_cem_exc, _cem_types)
            except TypeError:
                _cem_result = False
            const_stack.insert(0, _cem_result)
            if __INTERNAL__DEBUG_LOG:
                log.write("DEBUG => CHECK_EXC_MATCH: %s\n" % _cem_result)

        elif opcode_value == 'POP_EXCEPT':
            # Pops the saved previous-exception placeholder pushed by PUSH_EXC_INFO.
            if const_stack:
                const_stack.pop(0)
            if __INTERNAL__DEBUG_LOG:
                log.write("DEBUG => POP_EXCEPT\n")

        elif opcode_value == 'COPY':
            # COPY(i): push a copy of STACK[-i] (1-indexed) to TOS.
            if arg > 0 and arg <= len(const_stack):
                const_stack.insert(0, const_stack[arg - 1])
            if __INTERNAL__DEBUG_LOG:
                log.write("DEBUG => COPY %d\n" % arg)

        elif opcode_value == 'RERAISE':
            # Re-raise the current exception from TOS (if it is one).
            if const_stack and isinstance(const_stack[0], BaseException):
                raise const_stack[0]
            if __INTERNAL__DEBUG_LOG:
                log.write("DEBUG => RERAISE\n")

        elif opcode_value == 'LOAD_ATTR':
            math0 = const_stack.pop(0)
            try:
                const_stack.insert(0, getattr(math0, bytecode.co_names[arg]))
            except Exception as _la_exc:
                _la_handled = False
                for _et_s, _et_e, _et_t, _et_d in _exc_table:
                    if _et_s <= _cur_offset < _et_e:
                        while len(const_stack) > _et_d:
                            const_stack.pop(0)
                        const_stack.insert(0, _la_exc)
                        i = offset_to_index.get(_et_t, i) - 1
                        _la_handled = True
                        break
                if not _la_handled:
                    log.write("ERROR => LOAD_ATTR %s failed: %s\n" % (bytecode.co_names[arg], _la_exc))
                    const_stack.insert(0, None)

        elif opcode_value == 'MAKE_FUNCTION':
            # Stack layout at MAKE_FUNCTION (from TOS down):
            #   TOS:   code object          (pushed last)
            #   TOS-1: closure tuple        (if bit 0x08)
            #   TOS-2: annotations dict     (if bit 0x04)
            #   TOS-3: kwonly defaults dict (if bit 0x02)
            #   TOS-4: positional defaults  (if bit 0x01, pushed first = deepest)
            _mf_code = const_stack.pop(0)   # code object is always at TOS
            _mf_closure = const_stack.pop(0) if arg & 0x08 else None
            if arg & 0x04: const_stack.pop(0)   # annotations dict
            if arg & 0x02: const_stack.pop(0)   # kwonly defaults dict
            # Capture positional defaults so the callable can apply them correctly.
            _mf_defs = const_stack.pop(0) if arg & 0x01 else ()
            const_stack.insert(0, _mf_make(_mf_code, globals_frame, _mf_defs, _mf_closure))

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
                result_stack, log = py2vm(func, [], log, fast_locals=fl, globals_frame=globals_frame)
                const_stack.insert(0, result_stack[0] if result_stack else None)

                if __INTERNAL__DEBUG_LOG:
                    log.write("DEBUG => called function %s(%s)\n" % (func.co_name, args_list))

            except AttributeError:
                try:
                    if func.__name__ == '__build_class__' and args_list:
                        _cb = args_list[0]
                        try:
                            # Locate _co and _gf by name in co_freevars to be
                            # robust against future changes to _mf_make's params.
                            _fv = list(_cb.__code__.co_freevars)
                            _cb_code = _cb.__closure__[_fv.index('_co')].cell_contents
                            _cb_gf = _cb.__closure__[_fv.index('_gf')].cell_contents
                            args_list[0] = py2vm.__class__(_cb_code, _cb_gf)
                        except Exception:
                            pass
                except AttributeError:
                    pass
                try:
                    result = func(*args_list, **kwargs)
                    const_stack.insert(0, result)
                except Exception as e:
                    log.write("ERROR => could not call %s: %s\n" % (func, e))
                    const_stack.insert(0, None)

        elif opcode_value == 'CALL':
            # Python 3.11 CALL opcode: like CALL_FUNCTION but stack also has a
            # NULL sentinel below the callable (pushed by PUSH_NULL).
            # KW_NAMES may have provided keyword-argument names for this call.
            argc = arg  # total argument count (positional + keyword)

            args_list = []
            count = argc
            while count > 0:
                args_list.insert(0, const_stack.pop(0))
                count -= 1

            func = const_stack.pop(0)
            # Pop the NULL sentinel pushed by PUSH_NULL (or self for method calls)
            if const_stack:
                const_stack.pop(0)

            # Split off keyword arguments if KW_NAMES was set.
            if _kw_names:
                _call_nkw = len(_kw_names)
                _call_kwargs = dict(zip(_kw_names, args_list[-_call_nkw:]))
                args_list = args_list[:-_call_nkw]
                _kw_names = ()
            else:
                _call_kwargs = {}

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
                result_stack, log = py2vm(func, [], log, fast_locals=fl, globals_frame=globals_frame)
                const_stack.insert(0, result_stack[0] if result_stack else None)

                if __INTERNAL__DEBUG_LOG:
                    log.write("DEBUG => CALL code object %s(%s)\n" % (func.co_name, args_list))

            except AttributeError:
                # Before calling natively, if this is __build_class__ unwrap any
                # _mf_callable wrapper in the first argument so the class body
                # runs as a real FunctionType (needed for proper class namespace).
                try:
                    if func.__name__ == '__build_class__' and args_list:
                        _cb = args_list[0]
                        try:
                            # Look up _co and _gf by name in co_freevars.
                            _fv = list(_cb.__code__.co_freevars)
                            _cb_code = _cb.__closure__[_fv.index('_co')].cell_contents
                            _cb_gf = _cb.__closure__[_fv.index('_gf')].cell_contents
                            args_list[0] = py2vm.__class__(_cb_code, _cb_gf)
                        except Exception:
                            pass
                except AttributeError:
                    pass
                try:
                    result = func(*args_list, **_call_kwargs)
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


_VM_EXEC_DEPTH = 0


def run_script(path):
    """Read a Python source file and execute it through the VM."""
    global _VM_EXEC_DEPTH
    if _VM_EXEC_DEPTH >= 1:
        return ''   # prevent infinite meta-circular recursion
    _VM_EXEC_DEPTH += 1
    try:
        _f = open(path)
        _src = _f.read()
        _f.close()
        return buildcode(_src)
    finally:
        _VM_EXEC_DEPTH -= 1


_sys = __import__('sys')
_argv_all = _sys.argv
_script = _argv_all[1] if _argv_all.__len__() > 1 else None

if _script is not None:
    # Shift sys.argv by one so the inner VM sees only one arg and
    # takes the else branch (running the built-in test) rather than
    # trying to recurse into run_script again.
    _sys.argv = _argv_all[1:]
    print(run_script(_script))
    _sys.argv = _argv_all  # restore
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
