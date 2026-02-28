# Py2VM — Python 3.11 bytecode interpreter with explicit frame stack.
# Targets CPython 3.11 unspecialized bytecode (Tiers 1-5).
# Generator/async opcodes (Tier 6) are explicitly rejected.

import dis as _dis_mod
import types as _types_mod
import weakref as _weakref_mod

# ---------------------------------------------------------------------------
# Sentinels
# ---------------------------------------------------------------------------
_NULL = object()   # PUSH_NULL sentinel — distinct from Python None
_UNSET = object()  # Uninitialized fast-local slot


# ---------------------------------------------------------------------------
# Comparison operator table (dis.cmp_op equivalent for COMPARE_OP)
# ---------------------------------------------------------------------------
CMP_OP = ('<', '<=', '==', '!=', '>', '>=', 'in', 'not in', 'is',
          'is not', 'exception match', 'BAD')


# ---------------------------------------------------------------------------
# Minimal string buffer (avoids importing io/StringIO)
# ---------------------------------------------------------------------------
class _StringIO:
    def __init__(self):
        self._buf = []

    def write(self, s):
        self._buf.append(s)

    def getvalue(self):
        return ''.join(self._buf)


# ---------------------------------------------------------------------------
# Bytecode decode cache (WeakKeyDictionary — auto-evicts when code object dies)
# ---------------------------------------------------------------------------
_DECODE_CACHE = _weakref_mod.WeakKeyDictionary()
_DECODE_HITS = 0
_DECODE_MISSES = 0


def _decode_uncached(code):
    """Build the full decode payload for a code object (not cached).

    Returns (instructions, offset_to_index, argvals, exc_table).
    instructions: list of (opname, arg, offset) tuples.
    offset_to_index: dict mapping byte offset -> instruction index.
    argvals: list of resolved argval per instruction (for jump targets).
    exc_table: list of (start, end, target, depth) tuples.
    """
    raw = list(_dis_mod.get_instructions(code, adaptive=False,
                                         show_caches=False))
    instructions = []
    offset_to_index = {}
    argvals = []
    for idx, instr in enumerate(raw):
        offset_to_index[instr.offset] = idx
        arg = instr.arg if instr.arg is not None else 0
        instructions.append((instr.opname, arg, instr.offset))
        argvals.append(instr.argval)
    exc_table = []
    if hasattr(code, 'co_exceptiontable') and code.co_exceptiontable:
        try:
            for entry in _dis_mod._parse_exception_table(code):
                exc_table.append((entry.start, entry.end, entry.target, entry.depth))
        except Exception:
            pass
    return (instructions, offset_to_index, argvals, exc_table)


def decode_cached(code):
    """Return cached (instructions, offset_to_index, argvals, exc_table)."""
    global _DECODE_HITS, _DECODE_MISSES
    hit = _DECODE_CACHE.get(code)
    if hit is not None:
        _DECODE_HITS += 1
        return hit
    _DECODE_MISSES += 1
    payload = _decode_uncached(code)
    _DECODE_CACHE[code] = payload
    return payload


def decode_cache_stats():
    """Return (hits, misses, current_size) for decode cache."""
    return (_DECODE_HITS, _DECODE_MISSES, len(_DECODE_CACHE))


def _get_builtins():
    """Return builtins as a dict."""
    try:
        return __builtins__ if isinstance(__builtins__, dict) else __builtins__.__dict__
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Frame — holds all per-invocation state
# ---------------------------------------------------------------------------
class Frame:
    __slots__ = (
        'code', 'ip', 'stack', 'locals_fast', 'cells', 'freevars',
        'globals', 'builtins', 'block_stack', 'name_dict',
        'kw_names', 'instructions', 'offset_to_index', 'argvals',
        'exc_table',
    )

    def __init__(self, code, globals_dict, builtins_dict, closure=None):
        self.code = code
        self.ip = 0
        self.stack = []
        self.locals_fast = [_UNSET] * code.co_nlocals
        # Cell vars: one mutable [value] cell per co_cellvars entry
        self.cells = [None] * len(code.co_cellvars)
        # Free vars: supplied by closure tuple
        if closure:
            self.freevars = list(closure)
        else:
            self.freevars = [None] * len(code.co_freevars)
        self.globals = globals_dict
        self.builtins = builtins_dict
        self.block_stack = []
        self.name_dict = {}  # module-scope names keyed by index
        self.kw_names = ()
        instructions, offset_to_index, argvals, exc_table = decode_cached(code)
        self.instructions = instructions
        self.offset_to_index = offset_to_index
        self.argvals = argvals
        self.exc_table = exc_table


# ---------------------------------------------------------------------------
# VMFunction — a guest-code callable (replaces host-recursive _mf_make)
# ---------------------------------------------------------------------------
class VMFunction:
    __slots__ = ('code', 'globals', 'defaults', 'kwdefaults',
                 'closure', 'name', 'annotations')

    def __init__(self, code, globals_dict, defaults=(), closure=None,
                 kwdefaults=None, annotations=None):
        self.code = code
        self.globals = globals_dict
        self.defaults = defaults
        self.closure = closure
        self.kwdefaults = kwdefaults
        self.annotations = annotations
        self.name = code.co_name


def _bind_args(frame, vmfunc, args, kwargs):
    """Populate frame.locals_fast from positional/keyword args + defaults."""
    co = vmfunc.code
    nparams = co.co_argcount
    defaults = vmfunc.defaults or ()
    n_defaults = len(defaults)
    first_default = nparams - n_defaults

    # Apply defaults first
    for i in range(n_defaults):
        frame.locals_fast[first_default + i] = defaults[i]

    # Apply keyword-only defaults
    if vmfunc.kwdefaults:
        kw_start = nparams
        for i in range(co.co_kwonlyargcount):
            name = co.co_varnames[kw_start + i]
            if name in vmfunc.kwdefaults:
                frame.locals_fast[kw_start + i] = vmfunc.kwdefaults[name]

    # Positional args
    for i in range(min(nparams, len(args))):
        frame.locals_fast[i] = args[i]

    # *args (CO_VARARGS = 0x04)
    if co.co_flags & 0x04:
        varargs_idx = nparams + co.co_kwonlyargcount
        frame.locals_fast[varargs_idx] = tuple(args[nparams:])
    # **kwargs (CO_VARKEYWORDS = 0x08)
    if co.co_flags & 0x08:
        varkw_idx = nparams + co.co_kwonlyargcount + (1 if co.co_flags & 0x04 else 0)
        own_kwargs = {}
        for k, v in kwargs.items():
            # Check if it's a named parameter
            try:
                idx = list(co.co_varnames[:nparams + co.co_kwonlyargcount]).index(k)
                frame.locals_fast[idx] = v
            except ValueError:
                own_kwargs[k] = v
        frame.locals_fast[varkw_idx] = own_kwargs
    else:
        # Spread keyword args into named parameters
        for k, v in kwargs.items():
            try:
                idx = list(co.co_varnames[:nparams + co.co_kwonlyargcount]).index(k)
                frame.locals_fast[idx] = v
            except ValueError:
                pass


# ---------------------------------------------------------------------------
# BINARY_OP dispatch table (Python 3.11)
# Regular: 0=+ 1=& 2=// 3=<< 5=* 6=% 7=| 8=** 9=>> 10=- 11=/ 12=^
# Inplace: add 13 to regular arg
# ---------------------------------------------------------------------------
def _binary_op(op_arg, a, b):
    """Execute BINARY_OP: a op b where a is TOS1 and b is TOS."""
    base = op_arg if op_arg < 13 else op_arg - 13
    if base == 0:  return a + b
    if base == 1:  return a & b
    if base == 2:  return a // b
    if base == 3:  return a << b
    if base == 5:  return a * b
    if base == 6:  return a % b
    if base == 7:  return a | b
    if base == 8:  return a ** b
    if base == 9:  return a >> b
    if base == 10: return a - b
    if base == 11: return a / b
    if base == 12: return a ^ b
    return NotImplemented


# ---------------------------------------------------------------------------
# Tier 6 rejection set
# ---------------------------------------------------------------------------
_TIER6_OPCODES = frozenset({
    'RETURN_GENERATOR', 'YIELD_VALUE', 'SEND', 'GET_YIELD_FROM_ITER',
    'GET_AWAITABLE', 'ASYNC_GEN_WRAP', 'BEFORE_ASYNC_WITH',
    'END_ASYNC_FOR', 'SETUP_ANNOTATIONS',
})


# ---------------------------------------------------------------------------
# Main interpreter
# ---------------------------------------------------------------------------
def py2vm(bytecode, stack=False, rec_log=False, fast_locals=None,
          globals_frame=None):

    if rec_log is not False and rec_log:
        log = rec_log
    else:
        log = _StringIO()
        log.write('py2vm output:\n')

    builtins = _get_builtins()
    globals_dict = (globals_frame if globals_frame is not None
                    else {'__builtins__': __builtins__})

    # Build initial frame
    frame0 = Frame(bytecode, globals_dict, builtins)
    if fast_locals:
        for name, val in fast_locals.items():
            if name == '__closure__':
                continue
            try:
                idx = list(frame0.code.co_varnames).index(name)
                frame0.locals_fast[idx] = val
            except ValueError:
                pass
        # Handle closure cells passed via fast_locals
        closure = fast_locals.get('__closure__')
        if closure is not None:
            for ci in range(min(len(closure), len(frame0.freevars))):
                frame0.freevars[ci] = closure[ci]

    frames = [frame0]
    final_retval = None

    while frames:
        f = frames[-1]
        if f.ip >= len(f.instructions):
            # Implicit return None at end of code
            frames.pop()
            if frames:
                frames[-1].stack.append(None)
            continue

        opname, arg, offset = f.instructions[f.ip]
        argval = f.argvals[f.ip]
        f.ip += 1
        stk = f.stack

        # ---------------------------------------------------------------
        # Tier 6 rejection
        # ---------------------------------------------------------------
        if opname in _TIER6_OPCODES:
            raise NotImplementedError(
                "Generator/async opcode not supported: %s" % opname)

        try:  # __exc_handler__

            # ---------------------------------------------------------------
            # NOP / RESUME / PRECALL
            # ---------------------------------------------------------------
            if opname == 'NOP' or opname == 'RESUME' or opname == 'PRECALL':
                pass

            # ---------------------------------------------------------------
            # Stack manipulation
            # ---------------------------------------------------------------
            elif opname == 'POP_TOP':
                if stk:
                    stk.pop()

            elif opname == 'PUSH_NULL':
                stk.append(_NULL)

            elif opname == 'SWAP':
                # SWAP(i): swap TOS with stack[-i] (1-indexed from TOS)
                if arg > 1 and len(stk) >= arg:
                    stk[-1], stk[-arg] = stk[-arg], stk[-1]

            elif opname == 'COPY':
                # COPY(i): push a copy of stack[-i]
                if arg > 0 and len(stk) >= arg:
                    stk.append(stk[-arg])

            elif opname == 'ROT_TWO':
                stk[-1], stk[-2] = stk[-2], stk[-1]

            elif opname == 'ROT_THREE':
                a = stk.pop()
                stk.insert(-2, a)

            elif opname == 'ROT_FOUR':
                a = stk.pop()
                stk.insert(-3, a)

            elif opname == 'DUP_TOP':
                stk.append(stk[-1])

            # ---------------------------------------------------------------
            # Load / Store constants and locals
            # ---------------------------------------------------------------
            elif opname == 'LOAD_CONST':
                stk.append(f.code.co_consts[arg])

            elif opname == 'LOAD_FAST':
                val = f.locals_fast[arg]
                if val is _UNSET:
                    stk.append(None)
                else:
                    stk.append(val)

            elif opname == 'STORE_FAST':
                f.locals_fast[arg] = stk.pop()

            elif opname == 'DELETE_FAST':
                f.locals_fast[arg] = _UNSET

            # ---------------------------------------------------------------
            # Name operations (module scope)
            # ---------------------------------------------------------------
            elif opname == 'LOAD_NAME':
                val = f.name_dict.get(arg)
                if val is None:
                    name_str = f.code.co_names[arg]
                    val = f.globals.get(name_str)
                    if val is None:
                        val = builtins.get(name_str)
                stk.append(val)

            elif opname == 'STORE_NAME':
                val = stk.pop()
                f.name_dict[arg] = val
                f.globals[f.code.co_names[arg]] = val

            elif opname == 'DELETE_NAME':
                f.name_dict.pop(arg, None)
                f.globals.pop(f.code.co_names[arg], None)

            # ---------------------------------------------------------------
            # Global operations
            # ---------------------------------------------------------------
            elif opname == 'LOAD_GLOBAL':
                # 3.11: arg = (name_index << 1) | push_null_flag
                name_idx = arg >> 1
                push_null = arg & 1
                name_str = f.code.co_names[name_idx]
                val = f.globals.get(name_str)
                if val is None:
                    val = builtins.get(name_str)
                if push_null:
                    stk.append(_NULL)
                stk.append(val)

            elif opname == 'STORE_GLOBAL':
                f.globals[f.code.co_names[arg]] = stk.pop()

            elif opname == 'DELETE_GLOBAL':
                f.globals.pop(f.code.co_names[arg], None)

            # ---------------------------------------------------------------
            # Attribute operations
            # ---------------------------------------------------------------
            elif opname == 'LOAD_ATTR':
                obj = stk.pop()
                stk.append(getattr(obj, f.code.co_names[arg]))

            elif opname == 'STORE_ATTR':
                obj = stk.pop()
                val = stk.pop()
                setattr(obj, f.code.co_names[arg], val)

            elif opname == 'DELETE_ATTR':
                obj = stk.pop()
                delattr(obj, f.code.co_names[arg])

            elif opname == 'LOAD_METHOD':
                # getattr returns a bound method, so push NULL + bound_method.
                # CALL will pop NULL and not prepend self (already bound).
                obj = stk.pop()
                attr = getattr(obj, f.code.co_names[arg])
                stk.append(_NULL)  # NULL sentinel (self already bound)
                stk.append(attr)   # bound method at TOS

            # ---------------------------------------------------------------
            # Unary operations (fixed: in-place replace, no stack leak)
            # ---------------------------------------------------------------
            elif opname == 'UNARY_POSITIVE':
                stk[-1] = +stk[-1]

            elif opname == 'UNARY_NEGATIVE':
                stk[-1] = -stk[-1]

            elif opname == 'UNARY_NOT':
                stk[-1] = not stk[-1]

            elif opname == 'UNARY_INVERT':
                stk[-1] = ~stk[-1]

            elif opname == 'GET_ITER':
                stk[-1] = iter(stk[-1])

            # ---------------------------------------------------------------
            # Binary operations
            # ---------------------------------------------------------------
            elif opname == 'BINARY_OP':
                b = stk.pop()
                a = stk.pop()
                stk.append(_binary_op(arg, a, b))

            elif opname == 'BINARY_SUBSCR':
                key = stk.pop()
                obj = stk.pop()
                stk.append(obj[key])

            elif opname == 'STORE_SUBSCR':
                key = stk.pop()
                obj = stk.pop()
                val = stk.pop()
                obj[key] = val

            elif opname == 'DELETE_SUBSCR':
                key = stk.pop()
                obj = stk.pop()
                del obj[key]

            # ---------------------------------------------------------------
            # Comparison
            # ---------------------------------------------------------------
            elif opname == 'COMPARE_OP':
                b = stk.pop()
                a = stk.pop()
                op_name = CMP_OP[arg]
                if   op_name == '<':      result = a < b
                elif op_name == '<=':     result = a <= b
                elif op_name == '==':     result = a == b
                elif op_name == '!=':     result = a != b
                elif op_name == '>':      result = a > b
                elif op_name == '>=':     result = a >= b
                elif op_name == 'in':     result = a in b
                elif op_name == 'not in': result = a not in b
                elif op_name == 'is':     result = a is b
                elif op_name == 'is not': result = a is not b
                else:                     result = False
                stk.append(result)

            elif opname == 'IS_OP':
                b = stk.pop()
                a = stk.pop()
                stk.append((a is not b) if arg else (a is b))

            elif opname == 'CONTAINS_OP':
                container = stk.pop()
                item = stk.pop()
                try:
                    stk.append((item not in container) if arg else (item in container))
                except TypeError:
                    stk.append(False)

            # ---------------------------------------------------------------
            # Jump operations
            # ---------------------------------------------------------------
            elif opname == 'JUMP_FORWARD':
                f.ip = f.offset_to_index[argval]

            elif opname == 'JUMP_BACKWARD':
                f.ip = f.offset_to_index[argval]

            elif opname == 'JUMP_ABSOLUTE':
                f.ip = f.offset_to_index[argval]

            elif opname == 'POP_JUMP_FORWARD_IF_FALSE':
                if not stk.pop():
                    f.ip = f.offset_to_index[argval]

            elif opname == 'POP_JUMP_FORWARD_IF_TRUE':
                if stk.pop():
                    f.ip = f.offset_to_index[argval]

            elif opname == 'POP_JUMP_BACKWARD_IF_FALSE':
                if not stk.pop():
                    f.ip = f.offset_to_index[argval]

            elif opname == 'POP_JUMP_BACKWARD_IF_TRUE':
                if stk.pop():
                    f.ip = f.offset_to_index[argval]

            elif opname == 'POP_JUMP_FORWARD_IF_NONE':
                if stk.pop() is None:
                    f.ip = f.offset_to_index[argval]

            elif opname == 'POP_JUMP_FORWARD_IF_NOT_NONE':
                if stk.pop() is not None:
                    f.ip = f.offset_to_index[argval]

            elif opname == 'POP_JUMP_IF_FALSE':
                if not stk.pop():
                    f.ip = f.offset_to_index[argval]

            elif opname == 'POP_JUMP_IF_TRUE':
                if stk.pop():
                    f.ip = f.offset_to_index[argval]

            elif opname == 'JUMP_IF_TRUE_OR_POP':
                if stk[-1]:
                    f.ip = f.offset_to_index[argval]
                else:
                    stk.pop()

            elif opname == 'JUMP_IF_FALSE_OR_POP':
                if not stk[-1]:
                    f.ip = f.offset_to_index[argval]
                else:
                    stk.pop()

            # ---------------------------------------------------------------
            # Iteration
            # ---------------------------------------------------------------
            elif opname == 'FOR_ITER':
                try:
                    stk.append(next(stk[-1]))
                except StopIteration:
                    stk.pop()  # pop iterator
                    f.ip = f.offset_to_index[argval]

            elif opname == 'END_FOR':
                stk.pop()  # last value
                stk.pop()  # iterator

            # ---------------------------------------------------------------
            # Build collections
            # ---------------------------------------------------------------
            elif opname == 'BUILD_LIST':
                if arg == 0:
                    stk.append([])
                else:
                    items = stk[-arg:]
                    del stk[-arg:]
                    stk.append(items)

            elif opname == 'BUILD_TUPLE':
                if arg == 0:
                    stk.append(())
                else:
                    items = tuple(stk[-arg:])
                    del stk[-arg:]
                    stk.append(items)

            elif opname == 'BUILD_SET':
                if arg == 0:
                    stk.append(set())
                else:
                    items = set(stk[-arg:])
                    del stk[-arg:]
                    stk.append(items)

            elif opname == 'BUILD_MAP':
                if arg == 0:
                    stk.append({})
                else:
                    d = {}
                    pairs = stk[-2*arg:]
                    del stk[-2*arg:]
                    for i in range(0, 2*arg, 2):
                        d[pairs[i]] = pairs[i+1]
                    stk.append(d)

            elif opname == 'BUILD_CONST_KEY_MAP':
                keys = stk.pop()
                vals = stk[-arg:]
                del stk[-arg:]
                stk.append(dict(zip(keys, vals)))

            elif opname == 'BUILD_SLICE':
                if arg == 3:
                    step = stk.pop()
                    stop = stk.pop()
                    start = stk.pop()
                    stk.append(slice(start, stop, step))
                else:
                    stop = stk.pop()
                    start = stk.pop()
                    stk.append(slice(start, stop))

            elif opname == 'BUILD_STRING':
                parts = stk[-arg:]
                del stk[-arg:]
                stk.append(''.join(str(p) for p in parts))

            elif opname == 'LIST_APPEND':
                val = stk.pop()
                stk[-arg].append(val)

            elif opname == 'LIST_EXTEND':
                items = stk.pop()
                stk[-arg].extend(items)

            elif opname == 'LIST_TO_TUPLE':
                stk[-1] = tuple(stk[-1])

            elif opname == 'SET_ADD':
                val = stk.pop()
                stk[-arg].add(val)

            elif opname == 'SET_UPDATE':
                items = stk.pop()
                stk[-arg].update(items)

            elif opname == 'MAP_ADD':
                val = stk.pop()
                key = stk.pop()
                stk[-arg][key] = val

            elif opname in ('DICT_UPDATE', 'DICT_MERGE'):
                other = stk.pop()
                stk[-arg].update(other)

            elif opname == 'UNPACK_SEQUENCE':
                seq = list(stk.pop())
                for item in reversed(seq):
                    stk.append(item)

            # ---------------------------------------------------------------
            # String formatting
            # ---------------------------------------------------------------
            elif opname == 'FORMAT_VALUE':
                have_spec = bool(arg & 0x04)
                conv = arg & 0x03
                spec = stk.pop() if have_spec else ''
                val = stk.pop()
                if conv == 1:   val = str(val)
                elif conv == 2: val = repr(val)
                elif conv == 3: val = ascii(val)
                stk.append(format(val, spec))

            # ---------------------------------------------------------------
            # Cell / closure operations
            # ---------------------------------------------------------------
            elif opname == 'MAKE_CELL':
                # Convert local at localsplus[arg] to a cell.
                # arg is co_varnames index for cellvars that are params.
                cell_idx = arg - f.code.co_nlocals if arg >= f.code.co_nlocals else None
                if cell_idx is not None and 0 <= cell_idx < len(f.cells):
                    f.cells[cell_idx] = [None]
                else:
                    # cellvar that is also a local parameter
                    init_val = f.locals_fast[arg] if arg < len(f.locals_fast) and f.locals_fast[arg] is not _UNSET else None
                    # Find which cellvar index this corresponds to
                    varname = f.code.co_varnames[arg] if arg < len(f.code.co_varnames) else ''
                    try:
                        ci = list(f.code.co_cellvars).index(varname)
                        f.cells[ci] = [init_val]
                    except ValueError:
                        pass

            elif opname == 'LOAD_CLOSURE':
                # Push cell for cellvar at localsplus[arg]
                nlocals = f.code.co_nlocals
                if arg >= nlocals:
                    ci = arg - nlocals
                    if ci < len(f.cells):
                        stk.append(f.cells[ci])
                    else:
                        stk.append([None])
                else:
                    varname = f.code.co_varnames[arg] if arg < len(f.code.co_varnames) else ''
                    try:
                        ci = list(f.code.co_cellvars).index(varname)
                        stk.append(f.cells[ci])
                    except ValueError:
                        stk.append([None])

            elif opname == 'LOAD_DEREF':
                nlocals = f.code.co_nlocals
                ncells = len(f.code.co_cellvars)
                if arg < nlocals + ncells:
                    ci = arg - nlocals
                    if 0 <= ci < len(f.cells):
                        cell = f.cells[ci]
                        stk.append(cell[0] if cell is not None else None)
                    else:
                        stk.append(None)
                else:
                    fi = arg - nlocals - ncells
                    if 0 <= fi < len(f.freevars):
                        cell = f.freevars[fi]
                        stk.append(cell[0] if cell is not None else None)
                    else:
                        stk.append(None)

            elif opname == 'STORE_DEREF':
                val = stk.pop()
                nlocals = f.code.co_nlocals
                ncells = len(f.code.co_cellvars)
                if arg < nlocals + ncells:
                    ci = arg - nlocals
                    if 0 <= ci < len(f.cells):
                        if f.cells[ci] is None:
                            f.cells[ci] = [None]
                        f.cells[ci][0] = val
                else:
                    fi = arg - nlocals - ncells
                    if 0 <= fi < len(f.freevars) and f.freevars[fi] is not None:
                        f.freevars[fi][0] = val

            elif opname == 'COPY_FREE_VARS':
                # Already handled by Frame.__init__ using closure param.
                # This is a hint that the first `arg` freevars come from closure.
                pass

            # ---------------------------------------------------------------
            # Exception handling
            # ---------------------------------------------------------------
            elif opname == 'PUSH_EXC_INFO':
                exc = stk.pop()
                stk.append(None)   # previous exception placeholder
                stk.append(exc)

            elif opname == 'CHECK_EXC_MATCH':
                exc_type = stk.pop()
                exc = stk[-1]  # stays on stack
                try:
                    stk.append(isinstance(exc, exc_type))
                except TypeError:
                    stk.append(False)

            elif opname == 'POP_EXCEPT':
                if stk:
                    stk.pop()

            elif opname == 'RERAISE':
                if stk and isinstance(stk[-1], BaseException):
                    raise stk[-1]

            elif opname == 'RAISE_VARARGS':
                if arg == 0:
                    raise  # re-raise current
                elif arg == 1:
                    raise stk.pop()
                elif arg == 2:
                    cause = stk.pop()
                    exc = stk.pop()
                    raise exc from cause

            elif opname == 'BEFORE_WITH':
                mgr = stk[-1]
                exit_method = mgr.__exit__
                stk[-1] = exit_method
                stk.append(mgr.__enter__())

            # ---------------------------------------------------------------
            # Import operations
            # ---------------------------------------------------------------
            elif opname == 'IMPORT_NAME':
                fromlist = stk.pop()
                level = stk.pop()
                name = f.code.co_names[arg]
                stk.append(__import__(name, f.globals, None, fromlist, level))

            elif opname == 'IMPORT_FROM':
                stk.append(getattr(stk[-1], f.code.co_names[arg]))

            elif opname == 'IMPORT_STAR':
                mod = stk.pop()
                attrs = getattr(mod, '__all__', None) or [k for k in dir(mod) if not k.startswith('_')]
                for k in attrs:
                    f.globals[k] = getattr(mod, k)

            # ---------------------------------------------------------------
            # LOAD_BUILD_CLASS
            # ---------------------------------------------------------------
            elif opname == 'LOAD_BUILD_CLASS':
                stk.append(builtins.get('__build_class__', __builtins__.__build_class__ if hasattr(__builtins__, '__build_class__') else None))

            # ---------------------------------------------------------------
            # KW_NAMES
            # ---------------------------------------------------------------
            elif opname == 'KW_NAMES':
                f.kw_names = f.code.co_consts[arg]

            # ---------------------------------------------------------------
            # MAKE_FUNCTION
            # ---------------------------------------------------------------
            elif opname == 'MAKE_FUNCTION':
                code_obj = stk.pop()
                closure = stk.pop() if arg & 0x08 else None
                ann = stk.pop() if arg & 0x04 else None
                kwdefaults = stk.pop() if arg & 0x02 else None
                defaults = stk.pop() if arg & 0x01 else ()
                stk.append(VMFunction(code_obj, f.globals, defaults, closure,
                                      kwdefaults, ann))

            # ---------------------------------------------------------------
            # CALL (3.11 protocol)
            # ---------------------------------------------------------------
            elif opname == 'CALL':
                argc = arg
                # CPython 3.11 CALL protocol:
                # Stack: [null_or_func, func_or_self, arg0, ..., argN-1]
                # PEEK(argc+2) determines method vs non-method call.
                # If PEEK(argc+2) is not _NULL → method call:
                #   func = PEEK(argc+2), self = PEEK(argc+1), prepend to args
                # If PEEK(argc+2) is _NULL → non-method call:
                #   func = PEEK(argc+1), discard NULL
                is_meth = False
                if len(stk) >= argc + 2:
                    if stk[-(argc + 2)] is not _NULL:
                        is_meth = True

                if is_meth:
                    args_list = list(stk[-argc:]) if argc > 0 else []
                    if argc > 0:
                        del stk[-argc:]
                    first_arg = stk.pop()  # self / implicit first arg
                    func = stk.pop()       # actual callable
                    args_list = [first_arg] + args_list
                else:
                    args_list = list(stk[-argc:]) if argc > 0 else []
                    if argc > 0:
                        del stk[-argc:]
                    func = stk.pop()       # callable
                    if stk:
                        stk.pop()          # _NULL sentinel

                # Split kwargs if KW_NAMES was set
                if f.kw_names:
                    nkw = len(f.kw_names)
                    call_kwargs = dict(zip(f.kw_names, args_list[-nkw:]))
                    args_list = args_list[:-nkw]
                    f.kw_names = ()
                else:
                    call_kwargs = {}

                if isinstance(func, VMFunction):
                    new_frame = Frame(func.code, func.globals, builtins,
                                      func.closure)
                    _bind_args(new_frame, func, args_list, call_kwargs)
                    frames.append(new_frame)
                    continue
                else:
                    # Native callable
                    # Special handling for __build_class__ with VMFunction body
                    if (func is builtins.get('__build_class__') and
                            args_list and isinstance(args_list[0], VMFunction)):
                        vmf = args_list[0]
                        args_list[0] = _types_mod.FunctionType(
                            vmf.code, vmf.globals)
                    stk.append(func(*args_list, **call_kwargs))

            # ---------------------------------------------------------------
            # CALL_FUNCTION (legacy, kept for compatibility)
            # ---------------------------------------------------------------
            elif opname == 'CALL_FUNCTION':
                argc_pos = arg & 0xff
                argc_kw = (arg >> 8) & 0xff
                kwargs = {}
                for _ in range(argc_kw):
                    val = stk.pop()
                    key = stk.pop()
                    kwargs[key] = val
                args_list = stk[-argc_pos:] if argc_pos > 0 else []
                if argc_pos > 0:
                    del stk[-argc_pos:]
                func = stk.pop()

                if isinstance(func, VMFunction):
                    new_frame = Frame(func.code, func.globals, builtins,
                                      func.closure)
                    _bind_args(new_frame, func, args_list, kwargs)
                    frames.append(new_frame)
                    continue
                else:
                    stk.append(func(*args_list, **kwargs))

            # ---------------------------------------------------------------
            # CALL_FUNCTION_EX
            # ---------------------------------------------------------------
            elif opname == 'CALL_FUNCTION_EX':
                call_kwargs = stk.pop() if (arg & 0x01) else {}
                call_args = stk.pop()
                func = stk.pop()
                # Pop NULL sentinel if present
                if stk and stk[-1] is _NULL:
                    stk.pop()
                if isinstance(func, VMFunction):
                    new_frame = Frame(func.code, func.globals, builtins,
                                      func.closure)
                    _bind_args(new_frame, func, tuple(call_args), call_kwargs)
                    frames.append(new_frame)
                    continue
                else:
                    stk.append(func(*call_args, **call_kwargs))

            # ---------------------------------------------------------------
            # RETURN_VALUE
            # ---------------------------------------------------------------
            elif opname == 'RETURN_VALUE':
                retval = stk.pop() if stk else None
                frames.pop()
                if frames:
                    frames[-1].stack.append(retval)
                else:
                    final_retval = retval
                continue

            # ---------------------------------------------------------------
            # Unknown opcode
            # ---------------------------------------------------------------
            else:
                log.write("unknown opcode: %s (arg=%s)\n" % (opname, arg))

        except Exception as _vm_exc:
            # General exception handler: check exception table
            _exc_handled = False
            for _et_s, _et_e, _et_t, _et_d in f.exc_table:
                if _et_s <= offset < _et_e:
                    while len(stk) > _et_d:
                        stk.pop()
                    stk.append(_vm_exc)
                    f.ip = f.offset_to_index.get(_et_t, f.ip)
                    _exc_handled = True
                    break
            if not _exc_handled:
                raise

    # Return value handling — maintain backward compatibility
    if stack is not False:
        if isinstance(stack, list):
            if final_retval is not None:
                stack.append(final_retval)
            return stack, log
        return stack, log
    else:
        return log.getvalue()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def buildcode(code):
    """Compile and execute a source string through the VM."""
    return py2vm(compile(code, '<none>', 'exec'))


_VM_EXEC_DEPTH = 0


def run_script(path):
    """Read a Python source file and execute it through the VM."""
    global _VM_EXEC_DEPTH
    if _VM_EXEC_DEPTH >= 1:
        return ''
    _VM_EXEC_DEPTH += 1
    try:
        with open(path) as fh:
            src = fh.read()
        return buildcode(src)
    finally:
        _VM_EXEC_DEPTH -= 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import sys as _sys
    _argv_all = _sys.argv
    _script = _argv_all[1] if len(_argv_all) > 1 else None

    if _script is not None:
        _sys.argv = _argv_all[1:]
        print(run_script(_script))
        _sys.argv = _argv_all
    else:
        code = """
def test(order):
    return order

print('This comes first')
print(test('second'))
print('and I am third')
"""
        print(buildcode(code))
