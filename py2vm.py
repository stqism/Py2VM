# Py2VM — Python 3.11 bytecode interpreter with explicit frame stack.
# Targets CPython 3.11 unspecialized bytecode (Tiers 1-6).
# Supports generators, coroutines, async generators, and all async opcodes.

import dis as _dis_mod
import types as _types_mod
import weakref as _weakref_mod

# ---------------------------------------------------------------------------
# Sentinels
# ---------------------------------------------------------------------------
_NULL = object()   # PUSH_NULL sentinel — distinct from Python None
_UNSET = object()  # Uninitialized fast-local slot
_YIELD = object()  # Signal from _run_frames: generator yielded


# ---------------------------------------------------------------------------
# Async generator wrapped value marker
# ---------------------------------------------------------------------------
class _AsyncGenWrappedValue:
    """Wraps a yielded value inside an async generator so the protocol
    can distinguish yields from returns."""
    __slots__ = ('value',)
    def __init__(self, value):
        self.value = value


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
                exc_table.append((entry.start, entry.end, entry.target, entry.depth,
                                  getattr(entry, 'lasti', False)))
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
# Generator / Coroutine / AsyncGenerator objects
# ---------------------------------------------------------------------------
class VMGenerator:
    """Generator object — wraps a suspended Frame for resumable execution."""

    def __init__(self, frame, builtins, log):
        self._frame = frame
        self._builtins = builtins
        self._log = log
        self._started = False
        self._closed = False

    def __iter__(self):
        return self

    def __next__(self):
        return self.send(None)

    def send(self, value):
        if self._closed:
            raise StopIteration
        if not self._started:
            if value is not None:
                raise TypeError(
                    "can't send non-None value to a just-started generator")
            self._started = True
        self._frame.stack.append(value)
        result, signal = _run_frames([self._frame], self._builtins, self._log)
        if signal is _YIELD:
            return result
        self._closed = True
        raise StopIteration(result)

    def throw(self, typ, val=None, tb=None):
        if self._closed:
            raise StopIteration
        exc = typ if isinstance(typ, BaseException) else (
            typ(val) if val is not None else typ())

        # If in yield-from: sub-iterator is on top of stack
        stk = self._frame.stack
        if stk and hasattr(stk[-1], 'throw'):
            sub = stk[-1]
            try:
                result = sub.throw(type(exc), exc)
                return result
            except StopIteration as e:
                stk.pop()
                stk.append(e.value)
                # Find SEND instruction to get its jump target
                for i in range(self._frame.ip - 1, -1, -1):
                    if self._frame.instructions[i][0] == 'SEND':
                        target = self._frame.argvals[i]
                        self._frame.ip = self._frame.offset_to_index[target]
                        break
                result, signal = _run_frames(
                    [self._frame], self._builtins, self._log)
                if signal is _YIELD:
                    return result
                self._closed = True
                raise StopIteration(result)
            except BaseException:
                stk.pop()
                raise

        # Normal throw: find exception handler via exception table
        f = self._frame
        # Offset of the yield point (instruction before current ip)
        yield_offset = 0
        for i in range(f.ip - 1, -1, -1):
            if f.instructions[i][0] in ('YIELD_VALUE', 'RETURN_GENERATOR'):
                yield_offset = f.instructions[i][2]
                break

        handled = False
        for entry in f.exc_table:
            start, end, target, depth = entry[:4]
            lasti = entry[4] if len(entry) > 4 else False
            if start <= yield_offset < end:
                while len(stk) > depth:
                    stk.pop()
                if lasti:
                    stk.append(yield_offset)
                stk.append(exc)
                f.ip = f.offset_to_index.get(target, f.ip)
                handled = True
                break

        if handled:
            result, signal = _run_frames(
                [self._frame], self._builtins, self._log)
            if signal is _YIELD:
                return result
            self._closed = True
            raise StopIteration(result)
        else:
            self._closed = True
            raise exc

    def close(self):
        if self._closed:
            return
        try:
            self.throw(GeneratorExit)
        except (GeneratorExit, StopIteration):
            self._closed = True
            return
        except BaseException:
            self._closed = True
            raise
        self._closed = True


class VMCoroutine:
    """Coroutine object — wraps a suspended Frame (async def)."""

    def __init__(self, frame, builtins, log):
        self._frame = frame
        self._builtins = builtins
        self._log = log
        self._started = False
        self._closed = False

    def __await__(self):
        return self

    def __iter__(self):
        return self

    def __next__(self):
        return self.send(None)

    def send(self, value):
        if self._closed:
            raise StopIteration
        if not self._started:
            if value is not None:
                raise TypeError(
                    "can't send non-None value to a just-started coroutine")
            self._started = True
        self._frame.stack.append(value)
        result, signal = _run_frames([self._frame], self._builtins, self._log)
        if signal is _YIELD:
            return result
        self._closed = True
        raise StopIteration(result)

    def throw(self, typ, val=None, tb=None):
        if self._closed:
            raise StopIteration
        exc = typ if isinstance(typ, BaseException) else (
            typ(val) if val is not None else typ())

        stk = self._frame.stack
        if stk and hasattr(stk[-1], 'throw'):
            sub = stk[-1]
            try:
                result = sub.throw(type(exc), exc)
                return result
            except StopIteration as e:
                stk.pop()
                stk.append(e.value)
                for i in range(self._frame.ip - 1, -1, -1):
                    if self._frame.instructions[i][0] == 'SEND':
                        target = self._frame.argvals[i]
                        self._frame.ip = self._frame.offset_to_index[target]
                        break
                result, signal = _run_frames(
                    [self._frame], self._builtins, self._log)
                if signal is _YIELD:
                    return result
                self._closed = True
                raise StopIteration(result)
            except BaseException:
                stk.pop()
                raise

        f = self._frame
        yield_offset = 0
        for i in range(f.ip - 1, -1, -1):
            if f.instructions[i][0] in ('YIELD_VALUE', 'RETURN_GENERATOR'):
                yield_offset = f.instructions[i][2]
                break

        handled = False
        for entry in f.exc_table:
            start, end, target, depth = entry[:4]
            lasti = entry[4] if len(entry) > 4 else False
            if start <= yield_offset < end:
                while len(stk) > depth:
                    stk.pop()
                if lasti:
                    stk.append(yield_offset)
                stk.append(exc)
                f.ip = f.offset_to_index.get(target, f.ip)
                handled = True
                break

        if handled:
            result, signal = _run_frames(
                [self._frame], self._builtins, self._log)
            if signal is _YIELD:
                return result
            self._closed = True
            raise StopIteration(result)
        else:
            self._closed = True
            raise exc

    def close(self):
        if self._closed:
            return
        try:
            self.throw(GeneratorExit)
        except (GeneratorExit, StopIteration):
            self._closed = True
            return
        except BaseException:
            self._closed = True
            raise
        self._closed = True


class VMAsyncGenASend:
    """Awaitable returned by async generator's __anext__/asend.
    Drives the async generator frame one step."""

    def __init__(self, async_gen, value):
        self._gen = async_gen
        self._value = value

    def __await__(self):
        return self

    def __iter__(self):
        return self

    def __next__(self):
        return self.send(None)

    def send(self, value):
        gen = self._gen
        if gen._closed:
            raise StopAsyncIteration
        if not gen._started:
            gen._started = True
            send_val = None
        else:
            send_val = value if self._value is None else self._value
            self._value = None  # only use once
        gen._frame.stack.append(send_val)
        result, signal = _run_frames(
            [gen._frame], gen._builtins, gen._log)
        if signal is _YIELD:
            if isinstance(result, _AsyncGenWrappedValue):
                # Async gen yielded — completes the __anext__ awaitable
                raise StopIteration(result.value)
            # Non-wrapped yield: re-yield for await delegation
            return result
        # Generator returned — means StopAsyncIteration
        gen._closed = True
        raise StopAsyncIteration

    def throw(self, typ, val=None, tb=None):
        return self._gen.athrow(typ, val, tb).send(None)


class VMAsyncGenAThrow:
    """Awaitable returned by async generator's athrow/aclose."""

    def __init__(self, async_gen, typ, val):
        self._gen = async_gen
        self._typ = typ
        self._val = val

    def __await__(self):
        return self

    def __iter__(self):
        return self

    def __next__(self):
        return self.send(None)

    def send(self, value):
        gen = self._gen
        if gen._closed:
            raise StopAsyncIteration
        exc = self._typ if isinstance(self._typ, BaseException) else (
            self._typ(self._val) if self._val is not None else self._typ())

        f = gen._frame
        stk = f.stack

        yield_offset = 0
        for i in range(f.ip - 1, -1, -1):
            if f.instructions[i][0] in ('YIELD_VALUE', 'RETURN_GENERATOR'):
                yield_offset = f.instructions[i][2]
                break

        handled = False
        for entry in f.exc_table:
            start, end, target, depth = entry[:4]
            lasti = entry[4] if len(entry) > 4 else False
            if start <= yield_offset < end:
                while len(stk) > depth:
                    stk.pop()
                if lasti:
                    stk.append(yield_offset)
                stk.append(exc)
                f.ip = f.offset_to_index.get(target, f.ip)
                handled = True
                break

        if handled:
            result, signal = _run_frames(
                [gen._frame], gen._builtins, gen._log)
            if signal is _YIELD:
                if isinstance(result, _AsyncGenWrappedValue):
                    raise StopIteration(result.value)
                return result
            gen._closed = True
            raise StopAsyncIteration
        else:
            gen._closed = True
            raise exc


class VMAsyncGenerator:
    """Async generator object — wraps a suspended Frame (async def + yield)."""

    def __init__(self, frame, builtins, log):
        self._frame = frame
        self._builtins = builtins
        self._log = log
        self._started = False
        self._closed = False

    def __aiter__(self):
        return self

    def __anext__(self):
        return VMAsyncGenASend(self, None)

    def asend(self, value):
        return VMAsyncGenASend(self, value)

    def athrow(self, typ, val=None, tb=None):
        return VMAsyncGenAThrow(self, typ, val)

    def aclose(self):
        return VMAsyncGenAThrow(self, GeneratorExit, None)


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
# Main interpreter loop (extracted for generator/coroutine reuse)
# ---------------------------------------------------------------------------
def _run_frames(frames, builtins, log):
    """Run the frame stack until return or yield.

    Returns (value, signal):
      signal is None  → normal return, value is the final return value
      signal is _YIELD → generator yielded, value is the yielded value
    """
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

            elif opname == 'JUMP_BACKWARD_NO_INTERRUPT':
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

            elif opname == 'POP_JUMP_BACKWARD_IF_NONE':
                if stk.pop() is None:
                    f.ip = f.offset_to_index[argval]

            elif opname == 'POP_JUMP_BACKWARD_IF_NOT_NONE':
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

            elif opname == 'UNPACK_EX':
                count_before = arg & 0xFF
                count_after = arg >> 8
                seq = list(stk.pop())
                if count_after:
                    starred = seq[count_before:-count_after]
                    after = seq[-count_after:]
                else:
                    starred = seq[count_before:]
                    after = []
                for a_item in reversed(after):
                    stk.append(a_item)
                stk.append(starred)
                for i in range(count_before - 1, -1, -1):
                    stk.append(seq[i])

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
                _deref_val = _UNSET
                if arg < nlocals + ncells:
                    ci = arg - nlocals
                    if 0 <= ci < len(f.cells):
                        cell = f.cells[ci]
                        _deref_val = cell[0] if cell is not None else _UNSET
                else:
                    fi = arg - nlocals - ncells
                    if 0 <= fi < len(f.freevars):
                        cell = f.freevars[fi]
                        _deref_val = cell[0] if cell is not None else _UNSET
                if _deref_val is _UNSET:
                    if arg < nlocals + ncells:
                        _dn = f.code.co_cellvars[arg - nlocals] if (arg - nlocals) < ncells else '?'
                    else:
                        _fi = arg - nlocals - ncells
                        _dn = f.code.co_freevars[_fi] if _fi < len(f.code.co_freevars) else '?'
                    raise NameError("free variable '%s' referenced before assignment in enclosing scope" % _dn)
                stk.append(_deref_val)

            elif opname == 'LOAD_CLASSDEREF':
                nlocals = f.code.co_nlocals
                ncells = len(f.code.co_cellvars)
                _cd_val = _UNSET
                # Try cell/free var first
                if arg < nlocals + ncells:
                    ci = arg - nlocals
                    if 0 <= ci < len(f.cells) and f.cells[ci] is not None:
                        _cd_val = f.cells[ci][0]
                else:
                    fi = arg - nlocals - ncells
                    if 0 <= fi < len(f.freevars) and f.freevars[fi] is not None:
                        _cd_val = f.freevars[fi][0]
                if _cd_val is _UNSET:
                    # Fallback: class namespace, then globals, then builtins
                    if arg < nlocals + ncells:
                        _cdn = f.code.co_cellvars[arg - nlocals] if (arg - nlocals) < ncells else ''
                    else:
                        _fi = arg - nlocals - ncells
                        _cdn = f.code.co_freevars[_fi] if _fi < len(f.code.co_freevars) else ''
                    _cd_val = f.globals.get(_cdn, _UNSET)
                    if _cd_val is _UNSET:
                        _cd_val = builtins.get(_cdn)
                stk.append(_cd_val)

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

            elif opname == 'DELETE_DEREF':
                nlocals = f.code.co_nlocals
                ncells = len(f.code.co_cellvars)
                if arg < nlocals + ncells:
                    ci = arg - nlocals
                    if 0 <= ci < len(f.cells):
                        if f.cells[ci] is None:
                            f.cells[ci] = [_UNSET]
                        else:
                            f.cells[ci][0] = _UNSET
                else:
                    fi = arg - nlocals - ncells
                    if 0 <= fi < len(f.freevars) and f.freevars[fi] is not None:
                        f.freevars[fi][0] = _UNSET

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

            elif opname == 'CHECK_EG_MATCH':
                _eg_type = stk.pop()
                _eg_exc = stk.pop()
                if hasattr(_eg_exc, 'split'):
                    _eg_match, _eg_rest = _eg_exc.split(_eg_type)
                    if _eg_match is None:
                        stk.append(_eg_exc)
                        stk.append(None)
                    else:
                        stk.append(_eg_rest)
                        stk.append(_eg_match)
                elif isinstance(_eg_exc, BaseException) and isinstance(_eg_exc, _eg_type):
                    stk.append(None)
                    stk.append(_eg_exc)
                else:
                    stk.append(_eg_exc)
                    stk.append(None)

            elif opname == 'POP_EXCEPT':
                if stk:
                    stk.pop()

            elif opname == 'RERAISE':
                if stk and isinstance(stk[-1], BaseException):
                    raise stk[-1]

            elif opname == 'PREP_RERAISE_STAR':
                _prs_exc = stk.pop()
                _prs_orig = stk.pop()
                if _prs_exc is None:
                    stk.append(None)
                else:
                    stk.append(_prs_exc)

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

            elif opname == 'WITH_EXCEPT_START':
                # Stack: [..., __exit__, lasti, prev_exc, exc]
                exc = stk[-1]
                exit_fn = stk[-4]
                if isinstance(exc, BaseException):
                    res = exit_fn(type(exc), exc, exc.__traceback__)
                else:
                    res = exit_fn(None, None, None)
                stk.append(res)

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

            elif opname == 'LOAD_ASSERTION_ERROR':
                stk.append(AssertionError)

            elif opname == 'PRINT_EXPR':
                _pval = stk.pop()
                if _pval is not None:
                    print(repr(_pval))

            # ---------------------------------------------------------------
            # Pattern matching (match/case)
            # ---------------------------------------------------------------
            elif opname == 'GET_LEN':
                stk.append(len(stk[-1]))

            elif opname == 'MATCH_MAPPING':
                from collections.abc import Mapping as _Mapping
                stk.append(isinstance(stk[-1], _Mapping))

            elif opname == 'MATCH_SEQUENCE':
                from collections.abc import Sequence as _Sequence
                _ms_val = stk[-1]
                stk.append(isinstance(_ms_val, _Sequence)
                           and not isinstance(_ms_val, (str, bytes, bytearray)))

            elif opname == 'MATCH_KEYS':
                _mk_keys = stk[-1]
                _mk_subj = stk[-2]
                try:
                    _mk_vals = []
                    for _mk_k in _mk_keys:
                        if _mk_k not in _mk_subj:
                            stk.append(None)
                            break
                        _mk_vals.append(_mk_subj[_mk_k])
                    else:
                        stk.append(tuple(_mk_vals))
                except (TypeError, KeyError):
                    stk.append(None)

            elif opname == 'MATCH_CLASS':
                _mc_kw = stk.pop()
                _mc_cls = stk.pop()
                _mc_subj = stk.pop()
                if not isinstance(_mc_subj, _mc_cls):
                    stk.append(None)
                else:
                    _mc_args = getattr(_mc_cls, '__match_args__', ())
                    try:
                        _mc_pos = []
                        for _mc_i in range(arg):
                            _mc_pos.append(getattr(_mc_subj, _mc_args[_mc_i]))
                        _mc_kwv = []
                        for _mc_an in _mc_kw:
                            _mc_kwv.append(getattr(_mc_subj, _mc_an))
                        stk.append(tuple(_mc_pos + _mc_kwv))
                    except (AttributeError, IndexError):
                        stk.append(None)

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
                        if vmf.closure:
                            _cls_closure = tuple(
                                (lambda _v: (lambda: _v).__closure__[0])(
                                    c[0] if c else None) for c in vmf.closure)
                            args_list[0] = _types_mod.FunctionType(
                                vmf.code, vmf.globals, vmf.name, None,
                                _cls_closure)
                        else:
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
            # Tier 6: Generator / Coroutine / Async opcodes
            # ---------------------------------------------------------------
            elif opname == 'RETURN_GENERATOR':
                gen_frame = frames.pop()
                flags = gen_frame.code.co_flags
                if flags & 0x200:        # CO_ASYNC_GENERATOR
                    obj = VMAsyncGenerator(gen_frame, builtins, log)
                elif flags & 0x80:       # CO_COROUTINE
                    obj = VMCoroutine(gen_frame, builtins, log)
                else:                    # CO_GENERATOR (0x20)
                    obj = VMGenerator(gen_frame, builtins, log)
                if frames:
                    frames[-1].stack.append(obj)
                else:
                    final_retval = obj
                continue

            elif opname == 'YIELD_VALUE':
                yielded = stk.pop()
                return (yielded, _YIELD)

            elif opname == 'SEND':
                value = stk.pop()       # sent value (TOS)
                sub_iter = stk[-1]      # sub-iterator (TOS1, peek)
                try:
                    if value is None:
                        result = next(sub_iter)
                    else:
                        result = sub_iter.send(value)
                    stk.append(result)   # falls through to YIELD_VALUE
                except StopIteration as _si:
                    stk.pop()            # remove sub-iterator
                    stk.append(_si.value)
                    f.ip = f.offset_to_index[argval]

            elif opname == 'GET_YIELD_FROM_ITER':
                iterable = stk[-1]
                if not isinstance(iterable,
                                  (VMGenerator, VMCoroutine)):
                    stk[-1] = iter(iterable)

            elif opname == 'GET_AWAITABLE':
                obj = stk[-1]
                if isinstance(obj, VMCoroutine):
                    pass  # already its own await-iterator
                elif hasattr(obj, '__await__'):
                    stk[-1] = obj.__await__()

            elif opname == 'ASYNC_GEN_WRAP':
                stk[-1] = _AsyncGenWrappedValue(stk[-1])

            elif opname == 'GET_AITER':
                stk[-1] = stk[-1].__aiter__()

            elif opname == 'GET_ANEXT':
                stk.append(stk[-1].__anext__())

            elif opname == 'BEFORE_ASYNC_WITH':
                mgr = stk[-1]
                exit_method = mgr.__aexit__
                stk[-1] = exit_method
                stk.append(mgr.__aenter__())

            elif opname == 'END_ASYNC_FOR':
                exc = stk.pop()
                stk.pop()  # remove async iterator
                if not isinstance(exc, StopAsyncIteration):
                    raise exc

            elif opname == 'SETUP_ANNOTATIONS':
                if '__annotations__' not in f.globals:
                    f.globals['__annotations__'] = {}

            # ---------------------------------------------------------------
            # Unknown opcode
            # ---------------------------------------------------------------
            else:
                log.write("unknown opcode: %s (arg=%s)\n" % (opname, arg))

        except Exception as _vm_exc:
            # General exception handler: check exception table
            _exc_handled = False
            for _et_entry in f.exc_table:
                _et_s, _et_e, _et_t, _et_d = _et_entry[:4]
                _et_lasti = _et_entry[4] if len(_et_entry) > 4 else False
                if _et_s <= offset < _et_e:
                    while len(stk) > _et_d:
                        stk.pop()
                    if _et_lasti:
                        stk.append(offset)
                    stk.append(_vm_exc)
                    f.ip = f.offset_to_index.get(_et_t, f.ip)
                    _exc_handled = True
                    break
            if not _exc_handled:
                raise

    return (final_retval, None)


# ---------------------------------------------------------------------------
# Main interpreter (thin wrapper around _run_frames)
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
        closure = fast_locals.get('__closure__')
        if closure is not None:
            for ci in range(min(len(closure), len(frame0.freevars))):
                frame0.freevars[ci] = closure[ci]

    frames = [frame0]
    final_retval, _signal = _run_frames(frames, builtins, log)

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


def vm_run(coro):
    """Minimal coroutine driver — run a coroutine to completion.

    Analogous to asyncio.run() but without real I/O scheduling.
    Works for coroutines that don't actually suspend on I/O.
    """
    value = None
    while True:
        try:
            value = coro.send(value)
        except StopIteration as e:
            return e.value


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
